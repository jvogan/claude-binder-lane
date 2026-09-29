#!/usr/bin/env python3
"""Turn binder-lane stage shards into Modal jobs on the Claude Science job surface.

The binder lane describes work as an argv array. Modal takes a shell command
string that Claude Science interpolates into a ``run.sh``. This module is the
translation between the two, plus the bookkeeping that translation forces.

Read this first, because the shape is not obvious.

**One job is one shard of one stage.** A shard is a homogeneous slice of one
stage's candidates. Every candidate in a shard shares one adapter, one
environment identity, and one resources block. That decision is settled in
``shard_design.md`` and this module implements it rather than revisiting it.

**Smoke is always its own job of one candidate.** It must finish before any
scale shard of the same stage is submitted. The reason is the timeout.
``run_timeout_s`` is per submit, so bundling the smoke call with thirty
siblings sizes the guard for thirty. A wedged single call would then hold a
GPU for the whole batch budget.

**Fan-out width is not known before the run.** A stage declares
``fanout.count_from`` pointing at an upstream artifact. The real width exists
only after that upstream stage finishes and its manifest is read. So the
dispatcher completes a stage, reads the produced artifact, counts it, and only
then decides how many shards the next stage needs. A resumed run redoes that
reading rather than trusting a recorded number.

**Two processes, two halves of this file.** Everything above
``# --- dispatch ---`` is pure planning. It reads files, computes shard splits,
and returns job specifications as plain dictionaries. It never touches Modal,
never costs money, and runs from an ordinary shell. Everything below needs the
``host`` object, which exists only inside the Claude Science ``repl`` kernel.
The command line therefore plans and reports. It cannot submit. Submitting is
done by importing this module inside a ``repl`` cell and calling
``submit_wave``, then ``collect_notifications`` on each batch of notifications,
then ``close_wave``. A wave is not finished until ``close_wave`` reports its
barrier met. Closing every handle is what commits the Volume the next wave
reads, so the close is the barrier rather than the cleanup after it.

Where the work lands
--------------------

A Modal sandbox is destroyed with its filesystem, so nothing survives a job
except two things. The first is a Modal Volume, mounted at a path chosen at
submit time, which is where the run bundle and every artifact live. The second
is ``./out/`` plus ``stdout.log`` and ``stderr.log``, which Claude Science tars
and harvests back into the workspace at ``hpc/<job_id>/`` on every outcome,
success and failure alike.

So each job writes twice. Bulk output goes to the Volume, because the next
stage reads it there. A small report goes to ``./out/``, because that is the
only copy the host can read without another container. Keep ``./out/`` under
roughly 100 MB compressed. A multi-gigabyte ``./out/`` risks ``harvest_failed``
even though the job succeeded.

What this module does not do
----------------------------

It does not select or invent adapter command paths. A profile with an unresolved
``command_argv_template``, parser, toolcheck, interpreter, or cache path is
refused before a paid submit.

It does not replace the local plan runner. ``execute_plan`` still calls the
local executor in ``lane.py``. This module supplies the Modal stage seam and its
host-side submit, harvest, and resume functions.

Sources
-------

``~/.claude-science/orgs/<org-uuid>/skills/remote-compute-modal/SKILL.md`` is
the compute API this module targets. Every API fact below cites a line in it.
Do not read ``~/.claude-science/skills/``, which is a stale cache describing a
compute API that no longer exists.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import re
import shlex
import tarfile
import textwrap
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence


# --- constants ---------------------------------------------------------------

# The GPU tiers the job surface accepts, with the VRAM each one carries.
# Source: SKILL.md "GPU tier reference", lines 624-630.
GPU_TIERS: tuple[tuple[str, int], ...] = (
    ("A10G", 24),
    ("A100", 40),
    ("A100-80GB", 80),
    ("H100", 80),
)

# The VRAM each accepted tier carries, keyed by tier name.
TIER_VRAM: dict[str, int] = dict(GPU_TIERS)

# What each shipped environment asks for when it runs on a GPU. This is the
# tier selection source. SKILL.md:633-637 says each bundled env carries a
# gpu_default in its META, that it is what the env's author sized for the
# typical workload, and to start there. Keys are env-file stems, which is what
# environment_identity carries. Values come from the META block of each file in
# the org skill tree's remote-compute-modal/envs/, cited per row.
SHIPPED_GPU_DEFAULTS: dict[str, str] = {
    "chemistry_gpu": "A100",  # chemistry_gpu.py:12
    "esmfold2_gpu": "A100-80GB",  # esmfold2_gpu.py:23
    "esmfold2_kit_gpu": "H100",  # envs/esmfold2_kit_gpu.py META
    "genie3_generator_gpu": "A100",  # envs/genie3_generator_gpu.py META
    "boltz2_kit_gpu": "H100",  # envs/boltz2_kit_gpu.py META
    "genomics_evo2_gpu": "H100",  # genomics_evo2_gpu.py:16
    "md_openmm_gpu": "A100",  # md_openmm_gpu.py:13
    "proteomics_boltz_gpu": "A100-80GB",  # proteomics_boltz_gpu.py:12
    "proteomics_gpu": "A100-80GB",  # proteomics_gpu.py:19
    "proteomics_jax_gpu": "A100",  # proteomics_jax_gpu.py:7
    "proteomics_openfold_gpu": "A100-80GB",  # proteomics_openfold_gpu.py:22
    "proteomics_rfd_diffdock_gpu": "A100",  # proteomics_rfd_diffdock_gpu.py:39
    "rfdiffusion_generator_gpu": "A100",  # envs/rfdiffusion_generator_gpu.py META
    "singlecell_gpu": "A100",  # singlecell_gpu.py:41
}

# The Protenix v2 environment is unknown here. No row exists for it because no
# shipped env installs Protenix; the string does not appear anywhere in the org
# skill tree. Anthropic's released multi-target prompt runs Protenix v2 bring-up
# on an H100 and times the layer-norm JIT compile there, so the envs/*.py someone writes should
# carry "gpu_default": "H100" and its stem belongs in this table on the same
# commit that registers it. Until then gpu_tier_for refuses the adapter, which
# is also what parse_environment_identity does with its __REQUIRED__ identity.

# Limits the job surface enforces. Each one is a real refusal, not a clamp.
INPUT_BYTES_CAP = 1024 * 1024 * 1024  # 1 GiB per submit, SKILL.md:317-318
INPUT_FILE_CAP = 64  # files per submit, SKILL.md:318-319
CONTAINER_TIMEOUT_CAP_S = 85_500  # 24 h platform life minus staging, SKILL.md:39-40
# The Modal provider row permits a user-selected concurrency of up to 10,000.
# This is a provider setting ceiling, not a campaign-science threshold.
MAXIMUM_PROVIDER_CONCURRENT_JOBS = 10_000

# Not one of those. The three caps above are the platform's, and this module
# refuses against them. This is a fallback for a setting the platform lets the
# user change: the per-provider concurrency of their Modal row, which defaults
# to 10 and which they raise to as much as 10,000 in Settings under Compute on
# the Modal provider page. A campaign that knows the user's value carries it in
# `provider.maximum_concurrent_jobs` and this number is never consulted.
# Track the setting rather than exceed it. The provider enforces concurrency at
# submit, so a wave wider than the row allows loses its surplus jobs to
# `error_kind='provider_concurrency_full'`.
# Source: references/compute-and-pacing.md:22-33.
DEFAULT_SCALE_WAVE_WIDTH = 10
# The generator environment owns this mount and Volume name. Both values are
# copied from envs/rfdiffusion_generator_gpu.py, which mirrors the installed
# RFdiffusion environment's /weights contract.
RFDIFFUSION_WEIGHT_VOLUME = {
    "mount": "/weights",
    "name": "claude-science-rfdiffusion-weights",
}
FINALIZE_MODULE = "claude_binder"

# The environment identity a Modal profile writes, for example
# "modal-env:esmfold2_gpu@spec_sha=0123abcd". The env name is the stem of a file
# in the skill's envs/ directory. The spec_sha is that file's content hash and
# comes back from build_env(). Both are scoped to one Modal Environment.
ENVIRONMENT_IDENTITY_RE = re.compile(
    r"^modal-env:(?P<env>[A-Za-z_][A-Za-z0-9_]*)@spec_sha=(?P<spec_sha>[0-9a-f]+)$"
)

# A resolved image reference. Claude Science validates an "im-" literal through
# Image.from_id(), which fails closed on a fabricated id (SKILL.md:398-400).
IMAGE_REF_RE = re.compile(r"^im-[0-9A-Za-z]+$")

# The token a profile writes where a value is not yet known. materialize refuses
# a bundle that still carries one, so seeing it here means the bundle is a draft.
UNRESOLVED = "__REQUIRED__"

# Markers the lane's own receipt validation requires on a smoke_scale stage.
# Source: claude_binder_lane.py validate_completed_receipt.
SMOKE_MARKER = "STAGE_SMOKE_PASSED"
SCALE_MARKER = "STAGE_SCALE_PASSED"

# What a phase validation is called, on the Volume and in the harvest. A smoke
# phase writes no canonical receipt, because the scale phase a receipt has to
# validate has not run, so this file is the only host-readable evidence that
# the smoke phase produced anything at all.
PHASE_VALIDATION = "phase-validation.json"

# Where the harvest leaves a job's returned files inside the workspace, under
# hpc/<job_id>/. Measured on three completed runs: every path the completion
# notification reported keeps the out/ prefix the job wrote to.
HARVEST_OUT_DIR = "out"

# Files this module writes into the run's artifact directory. Neither exists in
# the lane today, and without them a resumed run cannot tell a shard that never
# ran from one whose job is still going.
JOB_REGISTER = "jobs.jsonl"
FANOUT_REGISTER = "fanout.jsonl"
RETRY_AUTHORIZATION_REGISTER = "retry-authorizations.jsonl"
DISPATCH_LOCK = ".dispatch.lock"

# Artifact types whose rows reference files that another stage wrote.
#
# A candidate manifest is a list of pointers. Each row names its sequence, its
# design pose and its source structure by path, and carries the sha256 of each,
# so the reference and the means to verify it travel together in one row.
# Shipping such a manifest without the files it names ships a list of names,
# which is what an earlier screen opened before it died on the first FASTA.
#
# The key is the receipt's ``artifact_type``, not its ``schema_path``. A resolved
# plan leaves ``schema_path`` null for some outputs, ``passing-candidates`` among
# them, while ``artifact_type`` is always populated.
#
# A field this registry does not name is data, not a reference. Scanning for
# path-shaped strings would make the transport interpret artifact contents, and
# would follow a path recorded for provenance as though it were a dependency.
# The third element is what the companion digest covers, which is not uniform and
# was measured on a real manifest rather than assumed. ``design_pose_sha256``
# and ``structure_sha256`` hash the file. ``sequence_sha256`` hashes the residue
# string the FASTA carries, so it never equals the FASTA file's digest and cannot
# verify the transfer. Treating all three alike would refuse every real run.
FILE_BYTES = "file-bytes"
SEQUENCE_TEXT = "sequence-text"

CANDIDATE_MANIFEST_REFERENCES: tuple[tuple[str, str, str], ...] = (
    ("sequence_path", "sequence_sha256", SEQUENCE_TEXT),
    ("design_pose_path", "design_pose_sha256", FILE_BYTES),
    ("structure_path", "structure_sha256", FILE_BYTES),
)

MANIFEST_REFERENCE_FIELDS: dict[str, tuple[tuple[str, str, str], ...]] = {
    "candidate-manifest": CANDIDATE_MANIFEST_REFERENCES,
    "backbone-candidate-manifest": CANDIDATE_MANIFEST_REFERENCES,
    "sequence-candidate-manifest": CANDIDATE_MANIFEST_REFERENCES,
    "normalized-candidate-manifest": CANDIDATE_MANIFEST_REFERENCES,
    "passing-candidate-manifest": CANDIDATE_MANIFEST_REFERENCES,
    "optimized-candidate-manifest": CANDIDATE_MANIFEST_REFERENCES,
    "rescore-candidate-manifest": CANDIDATE_MANIFEST_REFERENCES,
    "target-msa-manifest": (("msa_path", "msa_sha256", FILE_BYTES),),
}

# Artifact return. A Modal job's ./out is the only thing that reaches the host,
# so a receipt-only return leaves every structure on the Volume and starves the
# first local stage that reads the run root. These name the return manifest and
# the content-addressed blobs beside it.
ARTIFACT_RETURN_MANIFEST = "artifact-return.json"
ARTIFACT_BLOB_PREFIX = "artifact-"
ARTIFACT_BLOB_SUFFIX = ".blob"
# Well under the roughly 100 MB compressed ./out ceiling, so the return itself
# cannot be what makes a successful job fail to harvest.
ARTIFACT_RETURN_MAX_BYTES = 64 * 1024 * 1024
# Only a phase that ends a stage has published outputs worth returning. A smoke
# phase is a rehearsal and a scale shard is merged by finalize before publish.
ARTIFACT_RETURN_PHASES = frozenset({"single", "finalize"})
ARTIFACT_STAGING_DIR = ".incoming-artifacts"

# The local executor writes this exact key set for every stage attempt. The
# remote receipt writer uses the set to prevent a second receipt dialect.
LOCAL_RECEIPT_KEYS = frozenset(
    {
        "schema_version",
        "stage_id",
        "adapter_id",
        "attempt_id",
        "stage_identity",
        "run_fingerprint",
        "started_at",
        "finished_at",
        "scale_count",
        "provider_calls",
        "shard_merges",
        "phase_results",
        "parser_results",
        "decision_validation",
        "summary_lineage_validation",
        "optimization_lineage_validation",
        "optimization_filter_validation",
        "optimization_measurement_validation",
        "optimization_selection_validation",
        "screen_pool_validation",
        "promotion_validation",
        "normalized_lineage_validation",
        "output_manifest",
        "errors",
        "ok",
    }
)

# These variables point at persistent caches in the environment recipe. A
# cache path without the corresponding Modal volume is a paid-stage refusal.
CACHE_ENVIRONMENT_KEYS = frozenset(
    {
        "HF_HOME",
        "HF_HUB_CACHE",
        "HUGGINGFACE_HUB_CACHE",
        "TORCH_HOME",
        "TORCH_EXTENSIONS_DIR",
        "XDG_CACHE_HOME",
        "TRANSFORMERS_CACHE",
    }
)
BOOTSTRAP_ARCHIVE = "bootstrap.tar.gz"
BOOTSTRAP_MANIFEST = "bootstrap-manifest.json"
BOOTSTRAP_SCRIPT = "bootstrap.sh"
RECEIPT_SCRIPT = "receipt.py"
PACKAGE_EXCLUDED_PARTS = frozenset(
    {"tests", "__pycache__", ".pytest_cache", ".claude-binder"}
)

# Environment variables the generated shard script exports. An adapter script
# reads these to learn which slice of the stage it owns. They exist because the
# lane's template tokens cannot express a shard: ALLOWED_TEMPLATE_TOKENS holds
# campaign_id, run_id, target_structure, target_chain, binder_chain, run_root,
# artifact_root, receipts_dir, attempt_id, attempt_dir, stage_id, config_path,
# plan_path, phase, count and optimization_round, and nothing else. Adding a
# shard token would be a change to the executor. Passing the slice through the
# environment is not.
SHARD_ENV_PREFIX = "CLAUDE_BINDER_LANE_"

FINALIZE_SPEC = """\
The finalize job does five things, in a container that mounts the run Volume with
the repository subset present. All five are built here.

Built, as claude_binder_lane.py merge-shards:

1. Merge every per-shard output tree under {attempt_dir}/{phase}/shards/NNNN/
   into the single tree the stage contract names. The contract names one file
   per phase, so parallel shards cannot write it directly. Records concatenate
   in shard index order, which reproduces the order an unsharded run would
   write. Files a glob output owns move into the phase directory, and every
   record field that named one is repointed.
2. Assert the merged record count equals count multiplied by the output's
   records_per_count, and refuse to write a file it already knows is short.
3. Call the lane's validate_stage_outputs for the phase and report its errors.

4. Run parser_result for the phase, call publish_stage_outputs on every non-smoke
   manifest, write {attempt_dir}/STAGE_SMOKE_PASSED or STAGE_SCALE_PASSED, and
   write the canonical receipt to {receipts_dir}/{stage_id}.json plus the
   identical copy at {attempt_dir}/stage-receipt.json. validate_completed_receipt
   compares the two and fails if they differ.

   Two orderings here are load bearing. The parser has to run after the merge,
   because its source_output_hashes have to match the merged file rather than the
   shards. And publication has to run before the manifest is hashed, because
   publish_stage_outputs writes published_path into the artifact record and the
   receipt hash covers it.
5. Copy the receipt into ./out/receipt.json so the harvest carries it back to
   the host. The host reads counts from that copy, because it cannot read the
   Volume from an ordinary process.

The lane functions used here are parser_result, publish_stage_outputs,
stage_identity, validate_completed_receipt, sha256_json and write_json, all in
lane.py.
"""


# --- small helpers -----------------------------------------------------------


def bounded_integer(
    value: Any,
    *,
    label: str,
    minimum: int,
    maximum: int | None = None,
) -> int:
    """Parse one configuration integer and refuse values outside its contract."""
    if isinstance(value, bool):
        raise ValueError(f"{label} must be an integer")
    try:
        resolved = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be an integer") from exc
    if isinstance(value, float) and not value.is_integer():
        raise ValueError(f"{label} must be an integer")
    if resolved < minimum or (maximum is not None and resolved > maximum):
        upper = f"..{maximum}" if maximum is not None else "+"
        raise ValueError(f"{label} must be within {minimum}{upper}; got {resolved}")
    return resolved


def paid_job_timeout_s(value: Any, *, label: str) -> int:
    """Validate a paid-job guard against the reviewed platform lifetime cap."""
    return bounded_integer(
        value,
        label=label,
        minimum=1,
        maximum=CONTAINER_TIMEOUT_CAP_S,
    )


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def load_json(path: Path) -> Any:
    return json.loads(Path(path).read_text())


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    text = Path(path).read_text()
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def write_json(path: Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")


@contextmanager
def dispatch_lock(run_root: Path):
    """Serialize reservation, retry, and settlement decisions for one run.

    The provider API cannot participate in a filesystem transaction. Holding
    this advisory lock from the final cap read through every submit/register
    append is the smallest honest critical section: a second kernel cannot see
    an unreserved budget or an unregistered sibling while the first submits.
    """
    path = Path(run_root) / "artifacts" / DISPATCH_LOCK
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def record_spend(*args: Any, **kwargs: Any) -> dict[str, Any]:
    """Write one charge through the package spend ledger implementation."""
    from claude_binder.lane import record_spend as lane_record_spend

    return lane_record_spend(*args, **kwargs)


def open_run_registers(*args: Any, **kwargs: Any) -> dict[str, str]:
    """Open the package registers before the first paid dispatch."""
    from claude_binder.lane import open_run_registers as lane_open_run_registers

    return lane_open_run_registers(*args, **kwargs)


def enforce_spend_cap(*args: Any, **kwargs: Any) -> dict[str, Any]:
    """Check the package spend cap before the dispatcher creates a handle."""
    from claude_binder.lane import enforce_spend_cap as lane_enforce_spend_cap

    return lane_enforce_spend_cap(*args, **kwargs)


def verify_execution_approval(
    plan: dict[str, Any], bundle_root: Path
) -> dict[str, Any]:
    """Check the package approval ledger through the installed lane module."""
    from claude_binder.lane import verify_execution_approval as lane_verify_execution_approval

    return lane_verify_execution_approval(plan, bundle_root)


def sha256_json(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def posix_join(*parts: str) -> str:
    """Join container paths. The container is Linux whatever the host is."""
    cleaned = [part.strip("/") for part in parts[1:]]
    head = parts[0].rstrip("/")
    return "/".join([head, *[part for part in cleaned if part]])


# --- adapter identity to Modal parameters ------------------------------------


def parse_environment_identity(value: str) -> dict[str, str]:
    """Split a profile's environment_identity into its env name and spec hash.

    Raises ValueError on anything else, including the unresolved placeholder.
    A wrong env name reaches Modal as a request for an image that does not
    exist, which is a slow and confusing failure. Failing here is faster.
    """
    match = ENVIRONMENT_IDENTITY_RE.match(str(value))
    if not match:
        raise ValueError(
            f"environment_identity is not a resolved Modal identity: {value!r}. "
            "The format is modal-env:<env-name>@spec_sha=<hash>, and both halves "
            "come from compute_details({provider: 'modal', mode: 'read'})."
        )
    return {"env": match.group("env"), "spec_sha": match.group("spec_sha")}


def gpu_tier_for(env_name: str, gpu_memory_gb: int | None = None) -> str:
    """Return the GPU tier to request for one adapter's environment.

    The tier comes from the environment's own ``gpu_default``, because
    SKILL.md:633-637 calls that what the env's author sized for the typical
    workload and says to start there.

    VRAM on its own cannot pick a tier. A100-80GB and H100 both carry 80 GB, so
    an ordered walk of the tier table by size stops at A100-80GB and never
    reaches H100. The same walk lands on A10G whenever an adapter is quiet about
    its appetite, and no shipped environment defaults to A10G.

    ``gpu_memory_gb`` only escalates. An adapter documented as needing more VRAM
    than its environment's default tier carries moves up to the smallest tier
    that holds it, which is the concrete reason SKILL.md:634-636 allows. A
    smaller number never downgrades the environment's own choice, because the
    env author sized that default for this workload and the adapter's hint is a
    floor rather than a ceiling.

    An unrecorded environment raises. Falling back to a tier would send Modal a
    job that OOMs on hardware nobody chose, and that reads as a science failure.
    """
    default = SHIPPED_GPU_DEFAULTS.get(env_name)
    if default is None:
        raise ValueError(
            f"environment {env_name!r} has no recorded gpu_default. Read the META "
            f"block of remote-compute-modal/envs/{env_name}.py and add its "
            "gpu_default to SHIPPED_GPU_DEFAULTS. Do not pick a tier here."
        )
    if gpu_memory_gb is None:
        return default
    wanted = int(gpu_memory_gb)
    if wanted <= TIER_VRAM[default]:
        return default
    for name, vram in GPU_TIERS:
        if vram >= wanted:
            return name
    largest = GPU_TIERS[-1]
    raise ValueError(
        f"no shipped Modal GPU tier carries {wanted} GB; the largest is "
        f"{largest[0]} at {largest[1]} GB"
    )


def provider_params(
    adapter: dict[str, Any],
    *,
    volumes: dict[str, str],
    container_timeout_s: int | None = None,
    cpu_only: bool = False,
) -> dict[str, Any]:
    """Translate one adapter's resources block into Modal's own kwarg names.

    ``provider_params`` is one flat dictionary. The accepted keys are image,
    env, gpu, cpu, memory, volumes and timeout, and Claude Science validates it
    at create() time, before anything is staged or billed (SKILL.md:207-209).
    The older family-nested {'modal': {...}} form is accepted as a mute alias,
    so the flat form is the one to write.

    Two units differ from the lane's. Modal's ``memory`` is MiB, while the
    adapter's ``memory_gb`` is GiB. Modal's ``gpu`` is a tier name string, while
    the adapter's ``gpu`` is a device count.
    """
    resources = adapter.get("resources")
    if not isinstance(resources, dict):
        raise ValueError(f"adapter {adapter.get('adapter_id')!r} has no resources block")

    identity = parse_environment_identity(adapter["environment_identity"])
    image = resources.get("container_image_digest")
    if not isinstance(image, str) or not IMAGE_REF_RE.match(image):
        raise ValueError(
            f"adapter {adapter.get('adapter_id')!r} has no resolved image reference. "
            "resources.container_image_digest holds the im-... id that build_env() "
            f"returned; it currently reads {image!r}."
        )

    params: dict[str, Any] = {
        "image": image,
        "env": identity["env"],
        "cpu": int(resources["cpu"]),
        "memory": int(resources["memory_gb"]) * 1024,
        "volumes": dict(volumes),
    }

    gpu_count = 0 if cpu_only else int(resources.get("gpu", 0))
    if gpu_count == 1:
        # gpu_memory_gb is optional in the adapter schema. Absent, take the
        # environment's own gpu_default rather than guessing the workload's
        # appetite. Present and larger than that tier, it escalates.
        requested = resources.get("gpu_memory_gb")
        params["gpu"] = gpu_tier_for(
            identity["env"],
            None if requested is None else int(requested),
        )
    elif gpu_count > 1:
        # The multi-GPU request form is unknown. SKILL.md's tier table names
        # single devices only and no line in the skill states how to request
        # more than one. The real answer is in Modal's own GPU documentation,
        # which SKILL.md defers to when it says provider_params uses Modal's
        # own kwarg names. Resolve it there before any adapter asks for gpu > 1.
        raise ValueError(
            f"adapter {adapter.get('adapter_id')!r} asks for {gpu_count} GPUs and the "
            "multi-GPU request form is not recorded in the Claude Science skill"
        )
    # gpu_count == 0 omits the key. Modal runs the container without a GPU, which
    # is why the environment-setup kernel has to reject gpu= explicitly.

    if container_timeout_s is not None:
        params["timeout"] = paid_job_timeout_s(
            container_timeout_s, label="container timeout"
        )
    # Omitted, the container timeout fills from Settings before the tier card
    # renders, so the card shows the real lifetime (SKILL.md:379-382).
    return params


# --- shard arithmetic --------------------------------------------------------


@dataclass(frozen=True)
class ShardSlice:
    """One contiguous slice of a stage's candidates."""

    index: int
    total: int
    start: int  # inclusive, zero based
    stop: int  # exclusive

    @property
    def width(self) -> int:
        return self.stop - self.start

    def as_dict(self) -> dict[str, int]:
        return {
            "shard_index": self.index,
            "shard_total": self.total,
            "start": self.start,
            "stop": self.stop,
            "width": self.width,
        }


def split_shards(count: int, target_width: int) -> list[ShardSlice]:
    """Split ``count`` candidates into slices of at most ``target_width``.

    Widths differ by at most one, so no shard is a straggler that sets the
    stage's wall clock on its own. A shard should run long enough to amortize
    the container start, which is 18 to 30 seconds cold on an A100, and short
    enough that ``run_timeout_s`` still has headroom on a slow candidate.
    """
    if count < 1:
        raise ValueError(f"cannot shard a nonpositive candidate count: {count}")
    if target_width < 1:
        raise ValueError(f"shard target width must be positive: {target_width}")
    shard_total = -(-count // target_width)  # ceiling division
    base, remainder = divmod(count, shard_total)
    slices: list[ShardSlice] = []
    cursor = 0
    for index in range(shard_total):
        width = base + (1 if index < remainder else 0)
        slices.append(ShardSlice(index=index, total=shard_total, start=cursor, stop=cursor + width))
        cursor += width
    return slices


# --- the dispatch policy -----------------------------------------------------


@dataclass
class DispatchPolicy:
    """Per-adapter knobs that have no home in the campaign schema.

    ``adapter.schema.json`` sets ``additionalProperties: false``, so a shard
    width cannot be written into an adapter record. This lives beside the run
    bundle as ``dispatch-policy.json`` instead.

    ``shard_width`` is candidates per scale shard. ``seconds_per_candidate``
    sizes ``run_timeout_s``.
    """

    shard_width: dict[str, int] = field(default_factory=dict)
    default_shard_width: int = 6
    seconds_per_candidate: dict[str, int] = field(default_factory=dict)
    startup_allowance_s: int = 300
    # The finalize job merges shard manifests and validates a phase. It is CPU
    # work whose cost tracks record count, not candidate wall clock, so it never
    # takes the adapter's per-candidate figure.
    finalize_timeout_s: int = 1800
    container_timeout_s: int | None = None
    volume_name: str | None = None
    volume_mount: str = "/data"
    # Values come from the selected Modal environment's build_env() result.
    # They are separate from the run Volume because the environment owns them.
    environment_volumes: dict[str, dict[str, str]] = field(default_factory=dict)
    # The lane uses this token for every installed-module invocation. A Modal
    # profile must resolve it before a paid submit.
    python_executable: str | None = None
    repo_dir: str = "/data/claude-binder-lane/repo"
    workspace_staging: str = "dispatch"
    # Repository-relative path of the program the finalize job runs, and the
    # subcommand it takes. The lane's own executor merges the shard output, because
    # the merge needs the stage contract, the token renderer and
    # validate_stage_outputs, all of which live in that file.
    finalize_script: str | None = FINALIZE_MODULE
    finalize_subcommand: str = "merge-shards"

    def __post_init__(self) -> None:
        """Reject unsafe policy values before they can influence a paid job."""
        bounded_integer(
            self.default_shard_width,
            label="default shard width",
            minimum=1,
        )
        for adapter_id, width in self.shard_width.items():
            bounded_integer(
                width,
                label=f"shard width for adapter {adapter_id!r}",
                minimum=1,
            )
        for adapter_id, seconds in self.seconds_per_candidate.items():
            paid_job_timeout_s(
                seconds,
                label=f"seconds per candidate for adapter {adapter_id!r}",
            )
        bounded_integer(
            self.startup_allowance_s,
            label="startup allowance",
            minimum=0,
            maximum=CONTAINER_TIMEOUT_CAP_S,
        )
        paid_job_timeout_s(self.finalize_timeout_s, label="finalize timeout")
        if self.container_timeout_s is not None:
            paid_job_timeout_s(self.container_timeout_s, label="container timeout")

    @classmethod
    def load(cls, path: Path | None) -> "DispatchPolicy":
        if path is None:
            return cls()
        raw = load_json(path)
        known = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        unknown = sorted(set(raw) - known)
        if unknown:
            raise ValueError(f"dispatch policy has unknown fields: {', '.join(unknown)}")
        return cls(**raw)

    def width_for(self, adapter_id: str, *, ceiling: int | None = None) -> int:
        """Return a shard width bounded by the approved fan-out when supplied."""
        width = self.shard_width.get(adapter_id, self.default_shard_width)
        if ceiling is None:
            return bounded_integer(
                width,
                label=f"shard width for adapter {adapter_id!r}",
                minimum=1,
            )
        maximum = bounded_integer(
            ceiling,
            label=f"fan-out ceiling for adapter {adapter_id!r}",
            minimum=1,
        )
        # The ceiling counts candidates, the width counts candidates per shard,
        # so a width above the ceiling is a batch with nothing left to fill it
        # rather than a contract violation. A width this adapter configured is
        # still refused, because an operator who names a width above the cap has
        # said two contradictory things. The default carries no such claim: it
        # knows nothing about how many candidates this run promoted, so it
        # clamps. `validate_scale_count` already holds the candidate count at or
        # below the ceiling, so the clamp cannot change the shard count and
        # cannot change what the stage dispatches.
        if adapter_id not in self.shard_width:
            return min(
                bounded_integer(
                    width,
                    label=f"shard width for adapter {adapter_id!r}",
                    minimum=1,
                ),
                maximum,
            )
        return bounded_integer(
            width,
            label=f"shard width for adapter {adapter_id!r}",
            minimum=1,
            maximum=maximum,
        )

    def run_timeout_s(self, adapter_id: str, *, candidates: int, stage_ceiling_s: int) -> int:
        """Size the per-submit job guard for one shard.

        This is the single most expensive unknown here. No compute job has ever
        run on this account, so no adapter has a measured per-candidate wall
        clock. Until one does, this returns the stage's whole
        timeout, which never truncates real work and costs the full stage budget
        on a hang.

        That fallback costs the smoke gate its whole point. A smoke job exists so
        that a wedged single call is stopped by a guard sized for one candidate
        rather than for thirty. An uncalibrated smoke gets the stage's ceiling
        instead, so it holds a GPU for as long as the batch would have.
        ``verify_plan`` names every adapter still in that state.

        Resolve it by reading ``res.wall_s`` off the first smoke job of each
        adapter, which is the seconds ``run.sh`` actually ran on the remote, then
        writing that number into ``dispatch-policy.json:seconds_per_candidate``.
        """
        stage_ceiling_s = paid_job_timeout_s(
            stage_ceiling_s, label=f"stage timeout for adapter {adapter_id!r}"
        )
        candidates = bounded_integer(
            candidates,
            label=f"candidate count for adapter {adapter_id!r}",
            minimum=1,
        )
        per_candidate = self.seconds_per_candidate.get(adapter_id)
        if per_candidate is None:
            return stage_ceiling_s
        per_candidate_s = paid_job_timeout_s(
            per_candidate, label=f"seconds per candidate for adapter {adapter_id!r}"
        )
        startup_allowance_s = bounded_integer(
            self.startup_allowance_s,
            label="startup allowance",
            minimum=0,
            maximum=CONTAINER_TIMEOUT_CAP_S,
        )
        sized = per_candidate_s * candidates + startup_allowance_s
        return paid_job_timeout_s(
            min(sized, stage_ceiling_s),
            label=f"sized job timeout for adapter {adapter_id!r}",
        )


# --- the container script ----------------------------------------------------


def shard_script(
    *,
    argv: Sequence[str],
    toolcheck_argv: Sequence[str] | None,
    parser_argv: Sequence[str],
    repo_dir: str,
    attempt_dir: str,
    shard: ShardSlice | None,
    phase: str,
    lifecycle_phase: str | None = None,
    stage_id: str,
    finalize_argv: Sequence[str] | None,
    receipt_argv: Sequence[str] | None,
    run_adapter: bool = True,
    run_toolcheck: bool = False,
    run_parser: bool = True,
    validate_phase: bool = False,
) -> str:
    """Build the bash the job runs.

    Claude Science interpolates ``command=`` into a ``run.sh`` that starts with
    ``set -euo pipefail`` and ``cd "$(dirname "$0")"``, then runs it under bash.
    SKILL.md says to keep ``command=`` to a single program with arguments and to
    ship anything more as a file, because multi-layer shell escaping is the most
    common cause of a syntax error at the remote. This function writes that
    file. The job's own ``command=`` is then a fixed ``bash shard.sh``, with no
    interpolation left to get wrong.

    Every value that reaches the script goes through ``shlex.quote``. There is
    one quoting layer here, not two, because the script is written as a file
    rather than nested inside a command string.
    """
    lifecycle_phase = lifecycle_phase or phase
    lines = [
        "#!/usr/bin/env bash",
        "# Generated by scripts/dispatch_modal.py. Do not edit by hand.",
        "set -euo pipefail",
        "",
        "mkdir -p out",
        'OPERON_OUT_DIR="$PWD/out"',
        f"mkdir -p {shlex.quote(attempt_dir)}",
        f"export PYTHONPATH={shlex.quote(posix_join(repo_dir, 'src'))}${{PYTHONPATH:+:$PYTHONPATH}}",
        f"cd {shlex.quote(repo_dir)}",
        "",
    ]

    exports = {
        f"{SHARD_ENV_PREFIX}STAGE_ID": stage_id,
        f"{SHARD_ENV_PREFIX}PHASE": phase,
        f"{SHARD_ENV_PREFIX}ATTEMPT_DIR": attempt_dir,
    }
    literal_exports = {f"{SHARD_ENV_PREFIX}OUT_DIR": '"$OPERON_OUT_DIR"'}
    if shard is not None:
        exports.update(
            {
                f"{SHARD_ENV_PREFIX}SHARD_INDEX": str(shard.index),
                f"{SHARD_ENV_PREFIX}SHARD_TOTAL": str(shard.total),
                f"{SHARD_ENV_PREFIX}SHARD_START": str(shard.start),
                f"{SHARD_ENV_PREFIX}SHARD_STOP": str(shard.stop),
                f"{SHARD_ENV_PREFIX}SHARD_WIDTH": str(shard.width),
                f"{SHARD_ENV_PREFIX}SHARD_MANIFEST": "/work/shard-manifest.json",
                f"{SHARD_ENV_PREFIX}SHARD_OUT_DIR": posix_join(
                    attempt_dir, phase, "shards", f"{shard.index:04d}"
                ),
            }
        )
    lines.append("# The slice this job owns. The lane's template tokens cannot")
    lines.append("# express a shard, so the adapter script reads it from here.")
    for key, value in sorted(exports.items()):
        lines.append(f"export {key}={shlex.quote(value)}")
    for key, value in sorted(literal_exports.items()):
        lines.append(f"export {key}={value}")
    lines.append("")

    if shard is not None:
        lines += [
            "# Per-candidate output must be complete before the next candidate starts.",
            "# Write to a temporary directory, then rename into place. A shard that",
            "# buffers every write until the end loses all of it to a timeout.",
            f'mkdir -p "${SHARD_ENV_PREFIX}SHARD_OUT_DIR"',
            "",
        ]

    receipt_prefix = shlex.join(receipt_argv) if receipt_argv else None
    if run_toolcheck:
        if not toolcheck_argv:
            raise ValueError("a job that runs toolcheck has no toolcheck argv")
        lines += [
            "echo \"dispatch_modal: running the adapter toolcheck\" >&2",
            "set +e",
            shlex.join(toolcheck_argv),
            "TOOLCHECK_RC=$?",
            "set -e",
            'if [ "$TOOLCHECK_RC" -ne 0 ]; then',
            '  echo "dispatch_modal: toolcheck failed" >&2',
        ]
        if receipt_prefix:
            lines.append(
                f"  {receipt_prefix} --failure-kind toolcheck --failure-returncode \"$TOOLCHECK_RC\" || true"
            )
        lines += [
            '  cp -f "' + posix_join(attempt_dir, "stage-receipt.json") + '" "$OPERON_OUT_DIR/receipt.json" 2>/dev/null || true',
            '  cp -f "' + posix_join(attempt_dir, "receipt-validation.json") + '" "$OPERON_OUT_DIR/receipt-validation.json" 2>/dev/null || true',
            '  exit "$TOOLCHECK_RC"',
            "fi",
            "",
        ]

    if run_adapter:
        lines += [
            "# {{count}} in the rendered argv is the phase total, because the stage",
            "# contract validates the merged phase against it. This job runs only the",
            "# candidates between SHARD_START and SHARD_STOP.",
            "echo \"dispatch_modal: running the adapter command\" >&2",
            "set +e",
            shlex.join(argv),
            "ADAPTER_RC=$?",
            "set -e",
            'if [ "$ADAPTER_RC" -ne 0 ]; then',
            '  echo "dispatch_modal: adapter command failed" >&2',
        ]
        if receipt_prefix:
            lines.append(
                f"  {receipt_prefix} --failure-kind command --failure-returncode \"$ADAPTER_RC\" || true"
            )
        lines += [
            '  cp -f "' + posix_join(attempt_dir, "stage-receipt.json") + '" "$OPERON_OUT_DIR/receipt.json" 2>/dev/null || true',
            '  cp -f "' + posix_join(attempt_dir, "receipt-validation.json") + '" "$OPERON_OUT_DIR/receipt-validation.json" 2>/dev/null || true',
            '  exit "$ADAPTER_RC"',
            "fi",
            "",
        ]
    else:
        lines += [
            "# A finalize job runs no adapter. Its whole purpose is to fold in what the",
            "# scale shards already wrote, and running the tool again would pay for the",
            "# same predictions a second time.",
            "",
        ]

    if finalize_argv is not None:
        lines += [
            "echo \"dispatch_modal: finalizing the stage\" >&2",
            "set +e",
            shlex.join(finalize_argv),
            "FINALIZE_RC=$?",
            "set -e",
            'if [ "$FINALIZE_RC" -ne 0 ]; then',
            '  echo "dispatch_modal: merge failed" >&2',
        ]
        if receipt_prefix:
            lines.append(
                f"  {receipt_prefix} --failure-kind finalize --failure-returncode \"$FINALIZE_RC\" || true"
            )
        lines += [
            '  cp -f "' + posix_join(attempt_dir, "stage-receipt.json") + '" "$OPERON_OUT_DIR/receipt.json" 2>/dev/null || true',
            '  cp -f "' + posix_join(attempt_dir, "receipt-validation.json") + '" "$OPERON_OUT_DIR/receipt-validation.json" 2>/dev/null || true',
            '  exit "$FINALIZE_RC"',
            "fi",
            "",
        ]

    if run_parser:
        lines += [
            "echo \"dispatch_modal: running the adapter parser\" >&2",
            "set +e",
            shlex.join(parser_argv),
            "PARSER_RC=$?",
            "set -e",
            'if [ "$PARSER_RC" -ne 0 ]; then',
            '  echo "dispatch_modal: parser failed" >&2',
        ]
        if receipt_prefix:
            lines.append(
                f"  {receipt_prefix} --failure-kind parser --failure-returncode \"$PARSER_RC\" || true"
            )
        lines += [
            '  cp -f "' + posix_join(attempt_dir, "stage-receipt.json") + '" "$OPERON_OUT_DIR/receipt.json" 2>/dev/null || true',
            '  cp -f "' + posix_join(attempt_dir, "receipt-validation.json") + '" "$OPERON_OUT_DIR/receipt-validation.json" 2>/dev/null || true',
            '  exit "$PARSER_RC"',
            "fi",
            "",
        ]

    if finalize_argv is not None:
        if not receipt_prefix:
            raise ValueError("finalize jobs need a receipt runner")
        lines += [
            "echo \"dispatch_modal: writing and validating the canonical receipt\" >&2",
            f"{receipt_prefix}",
            "",
        ]

    if validate_phase:
        if not receipt_prefix:
            raise ValueError("phase validation needs a receipt runner")
        lines += [
            "echo \"dispatch_modal: validating the phase contract\" >&2",
            f"{receipt_prefix} --validate-phase {shlex.quote(phase)}",
            "",
            "# The line above is the whole of a passing smoke phase, and it writes no",
            "# canonical receipt. Without this copy the harvest is empty, the job still",
            "# exits 0, and a gate that reads the exit code authorizes the paid scale",
            "# wave with nothing to read. The copy is deliberately intolerant: the",
            "# validation exists whenever the line above returned, so a failure here is",
            "# a real transport failure and the job should carry it.",
            "cp -f "
            + shlex.quote(posix_join(attempt_dir, phase, PHASE_VALIDATION))
            + f' "$OPERON_OUT_DIR/{PHASE_VALIDATION}"',
            "",
        ]

    if lifecycle_phase == "smoke" and run_parser:
        lines += [
            f"touch {shlex.quote(posix_join(attempt_dir, SMOKE_MARKER))}",
            "",
        ]

    lines += [
        "# What the host can read without another container. Keep it small. Roughly",
        "# 100 MB compressed under ./out/ is the ceiling, and past it the harvest can",
        "# give up even though the job succeeded. These two are tolerant because a",
        "# smoke job legitimately has neither; its evidence was copied above.",
        "cp -f "
        + shlex.quote(posix_join(attempt_dir, "stage-receipt.json"))
        + ' "$OPERON_OUT_DIR/receipt.json" 2>/dev/null || true',
        "cp -f "
        + shlex.quote(posix_join(attempt_dir, "receipt-validation.json"))
        + ' "$OPERON_OUT_DIR/receipt-validation.json" 2>/dev/null || true',
        "",
    ]
    if receipt_prefix and lifecycle_phase in ARTIFACT_RETURN_PHASES:
        lines += [
            "# The receipt alone leaves every structure on the Volume, which is a",
            "# different store from the host run root. A local stage downstream of this",
            "# one reads the run root and would find nothing. Return the declared files",
            "# too, content-addressed, so the host can verify and promote them.",
            f"{receipt_prefix} --return-artifacts \"$OPERON_OUT_DIR\""
            f" --return-max-bytes {ARTIFACT_RETURN_MAX_BYTES}",
            "",
        ]
    return "\n".join(lines) + "\n"


# --- job specifications ------------------------------------------------------


@dataclass
class JobSpec:
    """Everything one Modal job needs, with no Modal object in sight.

    This is a plain record on purpose. It can be written to disk, reviewed, and
    diffed before anything is submitted, and the submitting half of this module
    is the only code that turns it into a container.
    """

    stage_id: str
    adapter_id: str
    phase: str  # "smoke", "scale", "single" or "finalize"
    attempt_id: str
    shard: ShardSlice | None
    intent: str
    command: str
    script_text: str
    inputs: list[dict[str, str]]
    outputs: list[str]
    run_timeout_s: int
    provider_params: dict[str, Any]
    # The scale width is distinct from the smoke job's count of one.
    scale_count: int = 1
    expected_shards: list[dict[str, int]] = field(default_factory=list)
    receipt_script_text: str = ""
    unresolved_paths: list[str] = field(default_factory=list)
    bootstrap_files: list[dict[str, Any]] = field(default_factory=list)
    # Volume-relative directory that the bootstrap owns and synchronizes to the
    # package entries in ``bootstrap_files``. Kept separate from the repository
    # root so a persistent Volume cannot retain a deleted package file forever.
    bootstrap_package_destination: str | None = None

    def as_dict(self) -> dict[str, Any]:
        record = {
            "stage_id": self.stage_id,
            "adapter_id": self.adapter_id,
            "phase": self.phase,
            "attempt_id": self.attempt_id,
            "shard": self.shard.as_dict() if self.shard else None,
            "intent": self.intent,
            "command": self.command,
            "inputs": self.inputs,
            "outputs": self.outputs,
            "run_timeout_s": self.run_timeout_s,
            "provider_params": self.provider_params,
            "scale_count": self.scale_count,
            "expected_shards": self.expected_shards,
            "unresolved_paths": self.unresolved_paths,
        }
        return record


def render_tokens(template: str, context: dict[str, Any]) -> str:
    """Substitute the lane's {{token}} form.

    This repeats the executor's own rendering rather than importing it, because
    the executor's module sets REPO_ROOT from its own file location and pulls in
    the whole campaign validator. The token set is the executor's; anything
    outside it fails validation upstream, before a plan is ever materialized.
    """
    pattern = re.compile(r"\{\{([a-z][a-z0-9_]*)\}\}")

    def replace(match: re.Match[str]) -> str:
        key = match.group(1)
        if key not in context:
            raise ValueError(f"command token has no value: {key}")
        return str(context[key])

    rendered = pattern.sub(replace, template)
    if "{{" in rendered or "}}" in rendered:
        raise ValueError("command contains an unresolved or malformed token")
    return rendered


def render_argv(template: Sequence[str], context: dict[str, Any]) -> list[str]:
    return [render_tokens(item, context) for item in template]


def receipt_runner_script() -> str:
    """Return the package-backed receipt and finalization runner.

    The source is staged with each job because ``lane.py`` is the protected
    owner of the receipt contract. The runner imports that owner in the
    container and calls its parser, publisher, and validator functions.
    """
    return textwrap.dedent(
        r'''
        #!/usr/bin/env python3
        from __future__ import annotations

        import argparse
        import json
        import sys
        from datetime import datetime, timezone
        from pathlib import Path

        from claude_binder.lane import (
            load_json,
            parser_result,
            publish_stage_outputs,
            sha256_file,
            sha256_json,
            stage_identity,
            stage_provider_calls,
            validate_completed_receipt,
            validate_stage_outputs,
            write_json,
        )


        def utc_now() -> str:
            return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


        def join_path(*parts: str) -> str:
            head = parts[0].rstrip("/")
            tails = [part.strip("/") for part in parts[1:]]
            return "/".join([head, *[part for part in tails if part]])


        def context_for(plan, stage, attempt_id, phase, count):
            runtime = plan["runtime"]
            run_root = str(runtime["run_root"]).rstrip("/")
            bundle_path = str(runtime["bundle_path"]).rstrip("/")
            artifact_root = join_path(run_root, "artifacts")
            context = dict(plan.get("context", {}))
            context.update(
                {
                    "run_root": run_root,
                    "artifact_root": artifact_root,
                    "receipts_dir": join_path(artifact_root, "receipts"),
                    "attempt_id": attempt_id,
                    "attempt_dir": join_path(
                        artifact_root, "stages", stage["stage_id"], "attempts", attempt_id
                    ),
                    "stage_id": stage["stage_id"],
                    "config_path": join_path(bundle_path, "config.resolved.json"),
                    "plan_path": join_path(bundle_path, "run-plan.json"),
                    "phase": phase,
                    "count": count,
                    "optimization_round": int(stage.get("optimization_round", 0)),
                }
            )
            return context


        def check_phase(stage, adapter, plan, attempt_dir, artifact_root, attempt_id, phase, count):
            context = context_for(plan, stage, attempt_id, phase, count)
            manifest = validate_stage_outputs(
                stage,
                context=context,
                attempt_dir=attempt_dir,
                artifact_root=artifact_root,
            )
            manifest["phase"] = phase
            for artifact in manifest["artifacts"]:
                artifact["phase"] = phase
            parsed = parser_result(adapter, context)
            parsed["phase"] = phase
            if parsed["ok"]:
                expected = sorted(
                    file_record["sha256"]
                    for artifact in manifest["artifacts"]
                    if artifact.get("declared_by") != "executor-derived"
                    for file_record in artifact["files"]
                )
                reported = sorted(str(value) for value in parsed["value"]["source_output_hashes"])
                if reported != expected:
                    parsed["ok"] = False
                    parsed["errors"].append(
                        "parser source_output_hashes do not match the stage outputs"
                    )
            errors = list(manifest["errors"]) + list(parsed["errors"])
            return manifest, parsed, errors


        def receipt_base(plan, stage, adapter, attempt_id, attempt_dir, receipts_dir, scale_count):
            identity = stage_identity(plan, stage, adapter, receipts_dir)
            return {
                "schema_version": 1,
                "stage_id": stage["stage_id"],
                "adapter_id": adapter["adapter_id"],
                "attempt_id": attempt_id,
                "stage_identity": identity,
                "run_fingerprint": plan["run_fingerprint"],
                "started_at": utc_now(),
                "finished_at": utc_now(),
                "scale_count": scale_count,
                "provider_calls": stage_provider_calls(stage, adapter, scale_count, []),
                "shard_merges": [],
                "phase_results": [],
                "parser_results": [],
                "decision_validation": None,
                "summary_lineage_validation": None,
                "optimization_lineage_validation": None,
                "optimization_filter_validation": None,
                "optimization_measurement_validation": None,
                "optimization_selection_validation": None,
                "screen_pool_validation": None,
                "promotion_validation": None,
                "normalized_lineage_validation": None,
                "output_manifest": {
                    "schema_version": 1,
                    "stage_id": stage["stage_id"],
                    "attempt_id": attempt_id,
                    "artifacts": [],
                    "output_manifest_sha256": sha256_json([]),
                },
                "errors": [],
                "ok": False,
            }


        def return_artifacts(plan, stage, attempt_dir, out_dir, max_bytes):
            """Copy the receipt's declared artifact files into ./out for the host.

            Modal returns only what a job writes under ./out. Everything else stays on
            the Volume, which is a different store from the host run root even when the
            two share a path string, so a local stage downstream of a Modal stage found
            nothing to read. This step closes that gap.

            The receipt is the manifest. It already records a sha256 and a byte count for
            every declared file, so this copies the bytes and lets the host verify them
            against hashes the receipt already carries. Each blob is named by its own
            hash, so a returned name cannot carry a path at all, let alone one that
            escapes the run root on extraction. Identical files return once.
            """
            receipt_path = attempt_dir / "stage-receipt.json"
            if not receipt_path.is_file():
                return {
                    "schema_version": 1,
                    "ok": False,
                    "complete": False,
                    "entries": [],
                    "errors": ["stage-receipt.json is missing"],
                }
            receipt = load_json(receipt_path)
            # The container reaches the run root through a mount, and resolving any path
            # under it yields the volume's own canonical prefix instead. Both spellings
            # name the same directory, and the receipt carries a mix: publish_stage_outputs
            # resolves published_path, and an adapter that resolves its own outputs records
            # the canonical form in path. Accept either root, so a genuine escape still
            # matches neither.
            run_root_source = Path(str(plan["runtime"]["run_root"]))
            run_root_text = str(run_root_source).rstrip("/")
            run_root_real = run_root_source.resolve()

            def run_root_relative(value):
                """Return value's run-root-relative form, or None when it is outside.

                The literal prefix is tried first, so every path that returned before
                still returns by the same route. Resolving both sides is the fallback,
                which is what admits the canonical spelling.
                """
                if run_root_text and value.startswith(run_root_text + "/"):
                    return value[len(run_root_text) + 1 :]
                try:
                    relative = Path(value).resolve().relative_to(run_root_real)
                except ValueError:
                    return None
                text = str(relative)
                return None if text in ("", ".") else text

            entries = []
            errors = []
            sizes = {}
            total = 0
            truncated = []

            def add_return_file(label, artifact_type, phase, source, destinations, evidence_kind=None):
                nonlocal total
                source = Path(source)
                if not source.is_file():
                    errors.append(f"{label}: declared file is missing: {source}")
                    return
                digest = sha256_file(source)
                actual = int(source.stat().st_size)
                if digest not in sizes:
                    if total + actual > max_bytes:
                        truncated.append(label)
                        return
                    (out_dir / f"artifact-{digest}.blob").write_bytes(source.read_bytes())
                    sizes[digest] = actual
                    total += actual
                entry = {
                    "artifact_id": label,
                    "artifact_type": artifact_type,
                    "phase": phase,
                    "sha256": digest,
                    "bytes": sizes[digest],
                    "destinations": sorted(set(destinations)),
                }
                if evidence_kind is not None:
                    entry["evidence_kind"] = evidence_kind
                entries.append(entry)

            for artifact in (receipt.get("output_manifest") or {}).get("artifacts") or []:
                label = artifact.get("artifact_id")
                for record in artifact.get("files") or []:
                    digest = record.get("sha256")
                    if not isinstance(digest, str) or len(digest) != 64:
                        errors.append(f"{label}: a file record carries no sha256")
                        continue
                    destinations = []
                    for key in ("path", "published_path"):
                        value = record.get(key)
                        if not isinstance(value, str) or not value:
                            continue
                        relative = run_root_relative(value)
                        if relative is None:
                            errors.append(f"{label}: {key} is outside the run root")
                            continue
                        destinations.append(relative)
                    if not destinations:
                        errors.append(f"{label}: no returnable path inside the run root")
                        continue
                    source = Path(str(record.get("path") or ""))
                    declared = record.get("bytes")
                    if source.is_file() and isinstance(declared, int) and declared != source.stat().st_size:
                        errors.append(
                            f"{label}: {source} is {source.stat().st_size} bytes, the receipt says {declared}"
                        )
                        continue
                    add_return_file(
                        label,
                        artifact.get("artifact_type"),
                        artifact.get("phase"),
                        source,
                        destinations,
                    )

            # validate_completed_receipt also binds executor evidence that is not a
            # scientific output artifact. Return it under the same receipt hash so the
            # host can validate the receipt in a genuinely separate store.
            for stored in receipt.get("parser_results") or []:
                parser_path = Path(str(stored.get("path") or ""))
                relative = run_root_relative(str(parser_path))
                if relative is None:
                    errors.append("parser-result: path is outside the run root")
                    continue
                add_return_file(
                    "executor-parser-result",
                    "executor-evidence",
                    stored.get("phase"),
                    parser_path,
                    [relative],
                    "parser-result",
                )
            attempt_receipt_relative = run_root_relative(str(receipt_path))
            if attempt_receipt_relative is None:
                errors.append("attempt-stage-receipt: path is outside the run root")
            else:
                add_return_file(
                    "executor-attempt-receipt",
                    "executor-evidence",
                    None,
                    receipt_path,
                    [attempt_receipt_relative],
                    "attempt-stage-receipt",
                )
            if stage.get("mode") == "smoke_scale":
                for marker in ("STAGE_SMOKE_PASSED", "STAGE_SCALE_PASSED"):
                    marker_path = attempt_dir / marker
                    relative = run_root_relative(str(marker_path))
                    if relative is None:
                        errors.append(f"{marker}: path is outside the run root")
                        continue
                    add_return_file(
                        f"executor-{marker.lower()}",
                        "executor-evidence",
                        None,
                        marker_path,
                        [relative],
                        "stage-marker",
                    )
            complete = not truncated and not errors
            if truncated:
                errors.append(
                    "artifact return exceeded "
                    + str(max_bytes)
                    + " bytes and stopped at "
                    + str(total)
                    + "; short artifacts: "
                    + ", ".join(sorted(set(str(item) for item in truncated)))
                )
            return {
                "schema_version": 1,
                "ok": complete,
                "complete": complete,
                "stage_id": receipt.get("stage_id"),
                "attempt_id": receipt.get("attempt_id"),
                "run_fingerprint": receipt.get("run_fingerprint"),
                "receipt_ok": bool(receipt.get("ok")),
                "receipt_sha256": sha256_file(receipt_path),
                "blob_count": len(sizes),
                "total_bytes": total,
                "max_bytes": max_bytes,
                "entries": entries,
                "errors": errors,
            }


        def main() -> int:
            parser = argparse.ArgumentParser()
            parser.add_argument("--plan", type=Path, required=True)
            parser.add_argument("--stage", required=True)
            parser.add_argument("--attempt-dir", type=Path, required=True)
            parser.add_argument("--receipts-dir", type=Path, required=True)
            parser.add_argument("--artifact-root", type=Path, required=True)
            parser.add_argument("--attempt-id", required=True)
            parser.add_argument("--scale-count", type=int, required=True)
            parser.add_argument("--failure-kind")
            parser.add_argument("--failure-returncode", type=int)
            parser.add_argument("--validate-phase", choices=["smoke", "scale", "single"])
            parser.add_argument("--return-artifacts", type=Path)
            parser.add_argument("--return-max-bytes", type=int, default=67108864)
            args = parser.parse_args()

            plan = load_json(args.plan)
            stage = next(item for item in plan["stages"] if item["stage_id"] == args.stage)
            adapter = next(item for item in plan["adapters"] if item["adapter_id"] == stage["adapter_id"])
            attempt_dir = args.attempt_dir.resolve()
            receipts_dir = args.receipts_dir.resolve()
            artifact_root = args.artifact_root.resolve()
            attempt_dir.mkdir(parents=True, exist_ok=True)
            receipts_dir.mkdir(parents=True, exist_ok=True)

            if args.return_artifacts:
                out_dir = args.return_artifacts.resolve()
                out_dir.mkdir(parents=True, exist_ok=True)
                manifest = return_artifacts(
                    plan, stage, attempt_dir, out_dir, args.return_max_bytes
                )
                write_json(out_dir / "artifact-return.json", manifest)
                # The host cannot complete the stage without these bytes. Make a
                # short or failed return part of the provider job outcome instead
                # of paying successfully and failing in a later local stage.
                return 0 if manifest.get("complete") else 1

            if args.validate_phase:
                count = 1 if args.validate_phase == "smoke" else args.scale_count
                manifest, parsed, errors = check_phase(
                    stage,
                    adapter,
                    plan,
                    attempt_dir,
                    artifact_root,
                    args.attempt_id,
                    args.validate_phase,
                    count,
                )
                # The identity fields are what let the host bind this file to the
                # job it came from. Without them a harvested validation is a record
                # of some phase somewhere, which cannot gate a paid wave.
                result = {
                    "schema_version": 1,
                    "ok": not errors,
                    "phase": args.validate_phase,
                    "stage_id": args.stage,
                    "attempt_id": args.attempt_id,
                    "run_fingerprint": plan["run_fingerprint"],
                    "manifest": manifest,
                    "parser_result": parsed,
                    "errors": errors,
                }
                write_json(attempt_dir / args.validate_phase / "phase-validation.json", result)
                if result["ok"] and args.validate_phase == "smoke":
                    (attempt_dir / "STAGE_SMOKE_PASSED").touch()
                return 0 if result["ok"] else 1

            receipt = receipt_base(
                plan,
                stage,
                adapter,
                args.attempt_id,
                attempt_dir,
                receipts_dir,
                args.scale_count,
            )
            if args.failure_kind:
                receipt["phase_results"] = [
                    {
                        "ok": False,
                        "phase": args.failure_kind,
                        "returncode": args.failure_returncode,
                    }
                ]
                receipt["errors"] = ["stage command failed"]
                receipt["finished_at"] = utc_now()
                write_json(attempt_dir / "stage-receipt.json", receipt)
                validation = validate_completed_receipt(receipt, stage, Path(plan["runtime"]["run_root"]))
                write_json(
                    attempt_dir / "receipt-validation.json",
                    {
                        "schema_version": 1,
                        "stage_id": args.stage,
                        "attempt_id": args.attempt_id,
                        "ok": False,
                        "errors": validation["errors"],
                        "receipt_sha256": sha256_file(attempt_dir / "stage-receipt.json"),
                    },
                )
                return 0

            phases = ["single"] if stage["mode"] == "single" else ["smoke", "scale"]
            output_manifests = []
            parser_results = []
            errors = []
            for phase in phases:
                count = 1 if phase == "smoke" else args.scale_count
                manifest, parsed, phase_errors = check_phase(
                    stage,
                    adapter,
                    plan,
                    attempt_dir,
                    artifact_root,
                    args.attempt_id,
                    phase,
                    count,
                )
                output_manifests.append(manifest)
                parser_results.append(parsed)
                errors.extend(f"phase {phase}: {error}" for error in phase_errors)
                merge_path = attempt_dir / phase / "merge-result.json"
                if merge_path.is_file():
                    receipt["shard_merges"].append(load_json(merge_path))
                else:
                    receipt["shard_merges"].append(
                        {"phase": phase, "ok": True, "mode": "direct", "errors": []}
                    )
            if not errors:
                publishable = [manifest for manifest in output_manifests if manifest["phase"] != "smoke"]
                errors.extend(publish_stage_outputs(publishable, artifact_root))
            combined = {
                "schema_version": 1,
                "stage_id": args.stage,
                "attempt_id": args.attempt_id,
                "artifacts": [
                    artifact
                    for manifest in output_manifests
                    for artifact in manifest["artifacts"]
                ],
            }
            combined["output_manifest_sha256"] = sha256_json(combined["artifacts"])
            write_json(attempt_dir / "stage-output-manifest.json", combined)
            receipt["finished_at"] = utc_now()
            receipt["phase_results"] = [{"ok": True, "phase": phase} for phase in phases]
            receipt["parser_results"] = parser_results
            receipt["provider_calls"] = stage_provider_calls(
                stage, adapter, args.scale_count, parser_results
            )
            receipt["output_manifest"] = combined
            receipt["errors"] = errors
            receipt["ok"] = not errors
            if receipt["ok"] and stage["mode"] == "smoke_scale":
                (attempt_dir / "STAGE_SCALE_PASSED").touch()
            write_json(attempt_dir / "stage-receipt.json", receipt)
            validation = validate_completed_receipt(receipt, stage, Path(plan["runtime"]["run_root"]))
            if not validation["ok"]:
                receipt["errors"].extend(validation["errors"])
                receipt["ok"] = False
                write_json(attempt_dir / "stage-receipt.json", receipt)
            else:
                write_json(receipts_dir / f"{args.stage}.json", receipt)
            write_json(
                attempt_dir / "receipt-validation.json",
                {
                    "schema_version": 1,
                    "stage_id": args.stage,
                    "attempt_id": args.attempt_id,
                    "ok": validation["ok"] and receipt["ok"],
                    "errors": validation["errors"],
                    "receipt_sha256": sha256_file(attempt_dir / "stage-receipt.json"),
                },
            )
            return 0 if receipt["ok"] else 1


        if __name__ == "__main__":
            raise SystemExit(main())
        '''
    )


def stage_context(
    plan: dict[str, Any],
    stage: dict[str, Any],
    *,
    attempt_id: str,
    phase: str,
    count: int,
) -> dict[str, Any]:
    """Build the token context for one phase of one stage.

    Every path here is a container path under the Volume mount, not a host path.
    The run bundle lives on the Volume, so the lane's own ``run_root`` and
    ``bundle_path`` are already Volume paths and need no rewriting.
    """
    runtime = plan["runtime"]
    run_root = str(runtime["run_root"]).rstrip("/")
    bundle_path = str(runtime["bundle_path"]).rstrip("/")
    artifact_root = posix_join(run_root, "artifacts")
    stage_id = stage["stage_id"]
    context = dict(plan.get("context", {}))
    context.update(
        {
            "run_root": run_root,
            "artifact_root": artifact_root,
            "receipts_dir": posix_join(artifact_root, "receipts"),
            "attempt_id": attempt_id,
            "attempt_dir": posix_join(
                artifact_root, "stages", stage_id, "attempts", attempt_id
            ),
            "stage_id": stage_id,
            "config_path": posix_join(bundle_path, "config.resolved.json"),
            "plan_path": posix_join(bundle_path, "run-plan.json"),
            "phase": phase,
            "count": count,
            "optimization_round": int(stage.get("optimization_round", 0)),
        }
    )
    return context


def volumes_for_adapter(adapter: dict[str, Any], policy: DispatchPolicy) -> dict[str, str]:
    """Return the run Volume plus every environment-owned read Volume.

    The environment recipes own cache mount names. The policy records the
    result of ``build_env()`` because the dispatcher cannot call that kernel
    function from a shell.
    """
    if policy.volume_name is None or UNRESOLVED in str(policy.volume_name):
        raise ValueError(
            "dispatch policy has no volume_name, so the run root has nowhere to live"
        )
    identity = parse_environment_identity(adapter["environment_identity"])
    volumes = {policy.volume_mount: policy.volume_name}
    for mount, name in policy.environment_volumes.get(identity["env"], {}).items():
        if not isinstance(mount, str) or not mount.startswith("/"):
            raise ValueError(f"environment volume mount is not absolute: {mount!r}")
        if not isinstance(name, str) or not name or UNRESOLVED in name:
            raise ValueError(f"environment volume name is empty at {mount}")
        existing = volumes.get(mount)
        if existing is not None and existing != name:
            raise ValueError(
                f"environment {identity['env']!r} maps {mount} to {name!r}, "
                f"but the run policy maps it to {existing!r}"
            )
        volumes[mount] = name
    if adapter.get("adapter_id") == "rfdiffusion-generator":
        mount = RFDIFFUSION_WEIGHT_VOLUME["mount"]
        name = RFDIFFUSION_WEIGHT_VOLUME["name"]
        existing = volumes.get(mount)
        if existing is not None and existing != name:
            raise ValueError(
                f"adapter rfdiffusion-generator needs {mount} on {name}, "
                f"but the dispatch policy assigns {existing}"
            )
        volumes[mount] = name
    missing_cache_mounts: list[str] = []
    environment = adapter.get("environment", {})
    if isinstance(environment, dict):
        for key, value in environment.items():
            if key not in CACHE_ENVIRONMENT_KEYS or not isinstance(value, str):
                continue
            if UNRESOLVED in value:
                missing_cache_mounts.append(f"{key}={value}")
                continue
            path = Path(value)
            if not path.is_absolute():
                missing_cache_mounts.append(f"{key}={value}")
                continue
            if not any(
                path == Path(mount) or Path(mount) in path.parents
                for mount in volumes
            ):
                missing_cache_mounts.append(f"{key}={value}")
    if missing_cache_mounts:
        raise ValueError(
            "cache paths have no attached Modal Volume: "
            + ", ".join(sorted(missing_cache_mounts))
        )
    return volumes


def build_job_spec(
    plan: dict[str, Any],
    stage: dict[str, Any],
    adapter: dict[str, Any],
    *,
    attempt_id: str,
    phase: str,
    count: int,
    shard: ShardSlice | None,
    policy: DispatchPolicy,
    finalize: bool,
    merge_phase: str | None = None,
    run_parser: bool = True,
    scale_count: int | None = None,
    expected_shards: Sequence[dict[str, int]] | None = None,
) -> JobSpec:
    """Turn one shard of one stage into a job specification.

    ``merge_phase`` is the phase a finalize job folds together. A finalize job runs
    after the scale shards, so its own phase name is ``finalize`` while the output
    tree it merges is ``scale``. Passing the wrong one sends the merge looking for
    an output directory that was never written.
    """
    # A finalize job addresses the phase directory its shards wrote, not one named
    # after itself, so the context it renders carries the merged phase.
    merge_phase = merge_phase or phase
    context = stage_context(
        plan, stage, attempt_id=attempt_id, phase=merge_phase, count=count
    )
    toolcheck_template = adapter.get("toolcheck_argv", [])
    if not isinstance(toolcheck_template, list):
        raise ValueError(f"adapter {adapter.get('adapter_id')!r} has no toolcheck argv list")
    templates = [
        *toolcheck_template,
        *adapter["command_argv_template"],
        *adapter["parser_argv_template"],
    ]
    uses_interpreter_token = any(
        "{{python_executable}}" in str(token) for token in templates
    )
    execution_python = policy.python_executable or (
        "python3" if not uses_interpreter_token else UNRESOLVED
    )
    if UNRESOLVED in execution_python:
        raise ValueError(
            f"stage {stage['stage_id']} has no resolved python_executable for its remote receipt and finalize runner"
        )
    context["python_executable"] = execution_python
    toolcheck_argv = render_argv(toolcheck_template, context)
    argv = render_argv(adapter["command_argv_template"], context)
    parser_argv = render_argv(adapter["parser_argv_template"], context)
    if any(UNRESOLVED in token for token in [*toolcheck_argv, *argv, *parser_argv]):
        raise ValueError(
            f"stage {stage['stage_id']} still carries an unresolved interpreter, "
            "toolcheck, command, or parser path. A paid Modal stage cannot submit "
            "while any execution path reads __REQUIRED__."
        )

    volumes = volumes_for_adapter(adapter, policy)
    resolved_scale_count = int(scale_count if scale_count is not None else count)
    receipt_argv = [
        execution_python,
        "/work/receipt.py",
        "--plan", context["plan_path"],
        "--stage", stage["stage_id"],
        "--attempt-dir", context["attempt_dir"],
        "--receipts-dir", context["receipts_dir"],
        "--artifact-root", context["artifact_root"],
        "--attempt-id", attempt_id,
        "--scale-count", str(resolved_scale_count),
    ]

    finalize_argv = None
    if finalize:
        if policy.finalize_script is None:
            raise NotImplementedError(
                "dispatch policy has no finalize_script.\n" + FINALIZE_SPEC
            )
        if policy.finalize_script == FINALIZE_MODULE:
            finalize_argv = [execution_python, "-m", FINALIZE_MODULE]
        else:
            finalize_argv = [
                execution_python,
                posix_join(policy.repo_dir, policy.finalize_script),
            ]
        finalize_argv.extend(
            [
                policy.finalize_subcommand,
                "--plan", context["plan_path"],
                "--stage", stage["stage_id"],
                "--attempt-dir", context["attempt_dir"],
                "--phase", merge_phase,
                "--count", str(count),
                "--json",
            ]
        )
        if expected_shards:
            finalize_argv.extend(["--expect-shards", "/work/expected-shards.json"])
        finalize_argv.extend(
            ["--out", posix_join(context["attempt_dir"], merge_phase, "merge-result.json")]
        )

    attempt_dir = context["attempt_dir"]
    script_text = shard_script(
        argv=argv,
        toolcheck_argv=toolcheck_argv,
        parser_argv=parser_argv,
        repo_dir=policy.repo_dir,
        attempt_dir=attempt_dir,
        shard=shard,
        phase=merge_phase,
        lifecycle_phase=phase,
        stage_id=stage["stage_id"],
        finalize_argv=finalize_argv,
        receipt_argv=receipt_argv,
        run_adapter=phase != "finalize",
        run_toolcheck=phase in {"smoke", "single"} and bool(toolcheck_argv),
        run_parser=run_parser,
        validate_phase=phase == "smoke",
    )

    stage_ceiling_s = paid_job_timeout_s(
        int(stage["timeout_minutes"]) * 60,
        label=f"stage {stage['stage_id']} timeout",
    )
    if phase == "finalize":
        run_timeout_s = min(
            paid_job_timeout_s(policy.finalize_timeout_s, label="finalize timeout"),
            stage_ceiling_s,
        )
    else:
        run_timeout_s = policy.run_timeout_s(
            adapter["adapter_id"],
            candidates=shard.width if shard else count,
            stage_ceiling_s=stage_ceiling_s,
        )
    run_timeout_s = paid_job_timeout_s(
        run_timeout_s,
        label=f"stage {stage['stage_id']} {phase} job timeout",
    )
    if policy.container_timeout_s is not None:
        container_timeout_s = paid_job_timeout_s(
            policy.container_timeout_s, label="container timeout"
        )
        if run_timeout_s > container_timeout_s:
            raise ValueError(
                f"stage {stage['stage_id']} {phase} job timeout {run_timeout_s} "
                f"exceeds configured container timeout {container_timeout_s}"
            )

    inputs = [{"src": "shard.sh", "dst": "shard.sh"}]
    inputs.append({"src": RECEIPT_SCRIPT, "dst": RECEIPT_SCRIPT})
    if shard is not None:
        inputs.append({"src": "shard-manifest.json", "dst": "shard-manifest.json"})
    if phase == "finalize" and expected_shards:
        inputs.append({"src": "expected-shards.json", "dst": "expected-shards.json"})
    if len(inputs) > INPUT_FILE_CAP:
        raise ValueError(
            f"a submit may carry at most {INPUT_FILE_CAP} input files; this one has {len(inputs)}"
        )

    return JobSpec(
        stage_id=stage["stage_id"],
        adapter_id=adapter["adapter_id"],
        phase=phase,
        attempt_id=attempt_id,
        shard=shard,
        intent=job_intent(plan, stage, phase=phase, count=count, shard=shard),
        command="bash shard.sh",
        script_text=script_text,
        inputs=inputs,
        # Omitting outputs brings back all of ./out/. Naming them keeps the
        # harvest to the small report and receipt, plus the content-addressed
        # artifact blobs the host promotes into its own run root.
        outputs=["*.json", "artifact-*.blob"],
        run_timeout_s=run_timeout_s,
        provider_params=provider_params(
            adapter,
            volumes=volumes,
            container_timeout_s=policy.container_timeout_s,
            # Merging records is CPU work. Holding the adapter's accelerator to
            # concatenate JSONL would bill a GPU hour for a file copy.
            cpu_only=phase == "finalize",
        ),
        scale_count=resolved_scale_count,
        expected_shards=[dict(item) for item in (expected_shards or [])],
        receipt_script_text=receipt_runner_script(),
        unresolved_paths=[],
    )


def job_intent(
    plan: dict[str, Any],
    stage: dict[str, Any],
    *,
    phase: str,
    count: int,
    shard: ShardSlice | None,
) -> str:
    """Write the one line about this job a person will actually see.

    The intent is the only per-job text that reaches the user, on the approval
    card and in the notification payload. An environment name tells a scientist
    nothing. A sentence naming the stage, the slice and the size does.
    """
    campaign = plan.get("campaign_id", "campaign")
    stage_id = stage["stage_id"]
    if phase == "finalize":
        return (
            f"{campaign}: {stage_id} merge, fold the scale shards into one output tree "
            f"of {count} candidate{'s' if count != 1 else ''}"
        )
    if phase == "smoke":
        return f"{campaign}: {stage_id} smoke, 1 candidate, gate before the scale shards"
    if shard is not None:
        return (
            f"{campaign}: {stage_id} shard {shard.index + 1} of {shard.total}, "
            f"candidates {shard.start + 1} to {shard.stop} of {count}"
        )
    return f"{campaign}: {stage_id}, {count} candidate{'s' if count != 1 else ''}"


# --- fan-out width, resolved at run time -------------------------------------


def harvested_receipt_path(
    workspace: Path, job_id: str, output_files: Sequence[str] | None = None
) -> Path:
    """Where the harvest left this job's receipt copy.

    Prefer the paths the completion notification reported. ``output_files`` is
    part of the ``compute_done`` payload and names what actually came back, so
    it is a fact rather than a guess.

    The constructed fallback used to guess a flat ``hpc/<job_id>/``, reading
    SKILL.md's "beside the deliverables" as dropping the ``out/`` prefix. Runs
    Three completed runs measured it: every returned path keeps the prefix, as
    ``hpc/<job_id>/out/<name>``. The harvest applies no name filter either. Those
    three runs brought back ``bootstrap-result.json``, ``artifact-return.json``,
    ``receipt.json``, ``receipt-validation.json`` and eleven content-addressed
    blobs, so any name a job writes under ``./out/`` comes home.
    """
    for candidate in output_files or []:
        if Path(candidate).name == "receipt.json":
            path = Path(candidate)
            return path if path.is_absolute() else workspace / path
    return workspace / "hpc" / job_id / HARVEST_OUT_DIR / "receipt.json"


def harvested_validation_path(
    workspace: Path, job_id: str, output_files: Sequence[str] | None = None
) -> Path:
    """Where the remote receipt-validation record landed after harvest."""
    for candidate in output_files or []:
        if Path(candidate).name == "receipt-validation.json":
            path = Path(candidate)
            return path if path.is_absolute() else workspace / path
    return workspace / "hpc" / job_id / HARVEST_OUT_DIR / "receipt-validation.json"


def validate_harvested_receipt(
    receipt: dict[str, Any],
    validation: dict[str, Any] | None,
    *,
    receipt_sha256: str | None = None,
) -> dict[str, Any]:
    """Validate the harvested receipt's identity and remote validation record."""
    errors: list[str] = []
    if set(receipt) != set(LOCAL_RECEIPT_KEYS):
        errors.append(
            "harvested receipt keys do not match the local receipt shape: "
            f"missing={sorted(LOCAL_RECEIPT_KEYS - set(receipt))} "
            f"extra={sorted(set(receipt) - LOCAL_RECEIPT_KEYS)}"
        )
    if receipt.get("ok") is not True:
        errors.append("harvested receipt is incomplete")
    if not isinstance(validation, dict):
        errors.append("harvested receipt-validation.json is missing")
    else:
        if validation.get("ok") is not True:
            errors.extend(str(error) for error in validation.get("errors", []))
        if validation.get("stage_id") != receipt.get("stage_id"):
            errors.append("harvested receipt validation names another stage")
        if validation.get("attempt_id") != receipt.get("attempt_id"):
            errors.append("harvested receipt validation names another attempt")
        if receipt_sha256 is not None and validation.get("receipt_sha256") != receipt_sha256:
            errors.append("harvested receipt validation hash does not match the receipt")
    return {
        "ok": not errors,
        "stage_id": receipt.get("stage_id"),
        "attempt_id": receipt.get("attempt_id"),
        "errors": errors,
    }


def harvested_phase_validation_path(
    workspace: Path, job_id: str, output_files: Sequence[str] | None = None
) -> Path:
    """Where the harvested phase validation landed."""
    for candidate in output_files or []:
        if Path(candidate).name == PHASE_VALIDATION:
            path = Path(candidate)
            return path if path.is_absolute() else workspace / path
    return workspace / "hpc" / job_id / HARVEST_OUT_DIR / PHASE_VALIDATION


def harvested_artifact_dir(
    workspace: Path, job_id: str, output_files: Sequence[str] | None = None
) -> Path:
    """Where the harvest left this job's artifact return."""
    for candidate in output_files or []:
        if Path(candidate).name == ARTIFACT_RETURN_MANIFEST:
            path = Path(candidate)
            path = path if path.is_absolute() else workspace / path
            return path.parent
    return workspace / "hpc" / job_id / HARVEST_OUT_DIR


def _declared_file_records(receipt: dict[str, Any]) -> list[dict[str, Any]]:
    manifest = receipt.get("output_manifest")
    artifacts = (manifest or {}).get("artifacts") or []
    return [
        record
        for artifact in artifacts
        for record in (artifact.get("files") or [])
        if isinstance(record, dict)
    ]


def promote_returned_artifacts(
    workspace: Path,
    row: dict[str, Any],
    *,
    run_root: Path,
    receipt: dict[str, Any],
    receipt_sha256: str | None = None,
    stage: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Verify a job's returned blobs and promote them into the host run root.

    The Volume and the host run root are separate stores that happen to share a
    path string, so nothing a Modal job writes reaches a later local stage unless
    it comes back through ``./out``. This is the host half of that transport.

    Every blob is hashed before it is written anywhere, and the hash must match
    the sha256 the receipt already recorded for that file. Every destination is
    resolved and required to stay inside the run root, so a receipt that names an
    escaping path is refused rather than followed. Promotion is atomic per file
    and idempotent: a destination already holding the right bytes is left alone,
    so a repeated harvest is a no-op.
    """
    job_id = str(row["job_id"])
    stage_id = receipt.get("stage_id")
    declared = _declared_file_records(receipt)
    attempt_id = str(receipt.get("attempt_id") or "")
    attempt_prefix = f"artifacts/stages/{stage_id}/attempts/{attempt_id}"

    def receipt_relative(value: Any) -> str | None:
        if not isinstance(value, str) or not value:
            return None
        parts = Path(value).parts
        try:
            index = parts.index("artifacts")
        except ValueError:
            return None
        relative = Path(*parts[index:]).as_posix()
        return relative if relative.startswith(attempt_prefix + "/") else None

    expected_evidence: dict[str, dict[str, Any]] = {}
    for stored in receipt.get("parser_results") or []:
        relative = receipt_relative(stored.get("path"))
        if relative is not None:
            expected_evidence[relative] = {
                "kind": "parser-result",
                "phase": stored.get("phase"),
                "value": stored.get("value"),
            }
    expected_evidence[f"{attempt_prefix}/stage-receipt.json"] = {
        "kind": "attempt-stage-receipt",
        "value": receipt,
    }
    if isinstance(stage, dict) and stage.get("mode") == "smoke_scale":
        for marker in ("STAGE_SMOKE_PASSED", "STAGE_SCALE_PASSED"):
            expected_evidence[f"{attempt_prefix}/{marker}"] = {
                "kind": "stage-marker",
                "value": b"",
            }
    result: dict[str, Any] = {
        "ok": True,
        "job_id": job_id,
        "stage_id": stage_id,
        "declared_files": len(declared),
        "promoted": [],
        "unchanged": [],
        "errors": [],
    }
    directory = harvested_artifact_dir(workspace, job_id, row.get("output_files"))
    manifest_path = directory / ARTIFACT_RETURN_MANIFEST
    if not manifest_path.is_file():
        result["ok"] = False
        result["errors"].append(
            f"the receipt requires {len(declared)} output files and "
            f"{len(expected_evidence)} executor evidence files but no "
            f"{ARTIFACT_RETURN_MANIFEST} came back: {manifest_path}"
        )
        return result
    try:
        manifest = load_json(manifest_path)
    except Exception as exc:
        result["ok"] = False
        result["errors"].append(
            f"{ARTIFACT_RETURN_MANIFEST} is unreadable: {type(exc).__name__}: {exc}"
        )
        return result
    if manifest.get("stage_id") != stage_id:
        result["ok"] = False
        result["errors"].append(
            f"{ARTIFACT_RETURN_MANIFEST} is for stage {manifest.get('stage_id')}, "
            f"not {stage_id}"
        )
        return result
    if manifest.get("attempt_id") != receipt.get("attempt_id"):
        result["ok"] = False
        result["errors"].append("the artifact return belongs to another attempt")
        return result
    if manifest.get("run_fingerprint") != receipt.get("run_fingerprint"):
        result["ok"] = False
        result["errors"].append("the artifact return belongs to another run")
        return result
    if manifest.get("receipt_ok") is not True:
        result["ok"] = False
        result["errors"].append("the artifact return was built from an incomplete receipt")
        return result
    if receipt_sha256 is not None and manifest.get("receipt_sha256") != receipt_sha256:
        result["ok"] = False
        result["errors"].append("the artifact return does not bind the harvested receipt")
        return result
    if not manifest.get("complete"):
        result["ok"] = False
        result["errors"].append(
            "the artifact return is incomplete: "
            + "; ".join(str(item) for item in (manifest.get("errors") or ["no reason given"]))
        )
        return result

    entries = manifest.get("entries") or []
    root = run_root.resolve()
    referenced: set[str] = set()
    planned: list[tuple[Path, str, bytes]] = []
    planned_destinations: set[Path] = set()
    returned_evidence: set[str] = set()
    for entry in entries:
        digest = entry.get("sha256")
        if not isinstance(digest, str) or len(digest) != 64 or not digest.isalnum():
            result["errors"].append(f"a return entry carries no usable sha256: {digest!r}")
            continue
        blob = directory / f"{ARTIFACT_BLOB_PREFIX}{digest}{ARTIFACT_BLOB_SUFFIX}"
        if not blob.is_file():
            result["errors"].append(f"{entry.get('artifact_id')}: blob is missing: {blob.name}")
            continue
        referenced.add(blob.name)
        actual = sha256_file(blob)
        if actual != digest:
            result["errors"].append(
                f"{entry.get('artifact_id')}: blob {blob.name} hashes to {actual}"
            )
            continue
        declared_bytes = entry.get("bytes")
        actual_bytes = blob.stat().st_size
        if isinstance(declared_bytes, int) and declared_bytes != actual_bytes:
            result["errors"].append(
                f"{entry.get('artifact_id')}: blob {blob.name} is {actual_bytes} bytes, "
                f"the manifest says {declared_bytes}"
            )
            continue
        payload = blob.read_bytes()
        destinations = entry.get("destinations") or []
        evidence_kind = entry.get("evidence_kind")
        if evidence_kind is not None:
            if entry.get("artifact_type") != "executor-evidence":
                result["errors"].append(
                    f"{entry.get('artifact_id')}: executor evidence has the wrong artifact type"
                )
                continue
            if len(destinations) != 1 or destinations[0] not in expected_evidence:
                result["errors"].append(
                    f"{entry.get('artifact_id')}: executor evidence destination is not required"
                )
                continue
            relative = destinations[0]
            expectation = expected_evidence[relative]
            if evidence_kind != expectation["kind"]:
                result["errors"].append(
                    f"{entry.get('artifact_id')}: executor evidence kind does not match {relative}"
                )
                continue
            if expectation["kind"] == "stage-marker":
                if payload != expectation["value"]:
                    result["errors"].append(f"{relative}: stage marker is not empty")
                    continue
            else:
                try:
                    evidence_value = json.loads(payload.decode("utf-8"))
                except Exception as exc:
                    result["errors"].append(
                        f"{relative}: executor evidence is not JSON: {type(exc).__name__}: {exc}"
                    )
                    continue
                if evidence_value != expectation["value"]:
                    result["errors"].append(
                        f"{relative}: executor evidence does not match the harvested receipt"
                    )
                    continue
            destination = (root / relative).resolve()
            if destination != root and root not in destination.parents:
                result["errors"].append(
                    f"{entry.get('artifact_id')}: destination escapes the run root: {relative}"
                )
                continue
            if destination in planned_destinations:
                result["errors"].append(f"duplicate artifact return destination: {relative}")
                continue
            planned_destinations.add(destination)
            planned.append((destination, relative, payload))
            returned_evidence.add(relative)
            continue
        matching_records = [
            record
            for artifact in (receipt.get("output_manifest") or {}).get("artifacts") or []
            if artifact.get("artifact_id") == entry.get("artifact_id")
            and artifact.get("artifact_type") == entry.get("artifact_type")
            for record in artifact.get("files") or []
            if record.get("sha256") == digest
            and record.get("bytes") in (None, declared_bytes)
        ]
        if not matching_records:
            result["errors"].append(
                f"{entry.get('artifact_id')}: return entry is not declared by the receipt"
            )
            continue
        source_paths = [
            value
            for record in matching_records
            for key in ("path", "published_path")
            if isinstance((value := record.get(key)), str) and value
        ]
        for source in source_paths:
            if not any(source == relative or source.endswith("/" + relative) for relative in destinations):
                result["errors"].append(
                    f"{entry.get('artifact_id')}: return omits receipt destination {source}"
                )
        for relative in destinations:
            if not isinstance(relative, str) or not relative:
                result["errors"].append(f"{entry.get('artifact_id')}: empty destination")
                continue
            candidate = Path(relative)
            if candidate.is_absolute():
                result["errors"].append(
                    f"{entry.get('artifact_id')}: destination is absolute: {relative}"
                )
                continue
            destination = (root / candidate).resolve()
            if destination != root and root not in destination.parents:
                result["errors"].append(
                    f"{entry.get('artifact_id')}: destination escapes the run root: {relative}"
                )
                continue
            if not any(
                source == relative or source.endswith("/" + relative)
                for source in source_paths
            ):
                result["errors"].append(
                    f"{entry.get('artifact_id')}: destination is not declared by the receipt: {relative}"
                )
                continue
            if destination in planned_destinations:
                result["errors"].append(f"duplicate artifact return destination: {relative}")
                continue
            planned_destinations.add(destination)
            planned.append((destination, relative, payload))

    extra = sorted(
        path.name
        for path in directory.glob(f"{ARTIFACT_BLOB_PREFIX}*{ARTIFACT_BLOB_SUFFIX}")
        if path.name not in referenced
    )
    if extra:
        result["errors"].append(
            "blobs came back that the manifest does not name: " + ", ".join(extra)
        )

    returned_hashes = {
        entry.get("sha256") for entry in entries if isinstance(entry.get("sha256"), str)
    }
    missing = sorted(
        {
            str(record.get("sha256"))
            for record in declared
            if isinstance(record.get("sha256"), str)
            and record.get("sha256") not in returned_hashes
        }
    )
    if missing:
        result["errors"].append(
            "the receipt declares files the return omits: " + ", ".join(missing)
        )
    missing_evidence = sorted(set(expected_evidence) - returned_evidence)
    if missing_evidence:
        result["errors"].append(
            "the receipt requires executor evidence the return omits: "
            + ", ".join(missing_evidence)
        )

    result["ok"] = not result["errors"]
    if not result["ok"]:
        return result
    for destination, relative, payload in planned:
        digest = hashlib.sha256(payload).hexdigest()
        if destination.is_file() and sha256_file(destination) == digest:
            result["unchanged"].append(relative)
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        staging = destination.parent / f".{destination.name}.{digest[:12]}.incoming"
        staging.write_bytes(payload)
        os.replace(staging, destination)
        result["promoted"].append(relative)
    return result


def rehome_provider_published_paths(
    receipt: dict[str, Any], *, run_root: Path, promotion: Any
) -> dict[str, Any]:
    """Re-point a provider receipt's published paths at the host copies.

    A receipt written inside a provider container records ``published_path`` as
    that container resolved it. Modal resolves a mounted Volume through its own
    canonical ``/__modal/volumes/<id>/`` root, so the recorded value names a real
    file under a name that does not exist on this host.
    ``validate_completed_receipt`` resolves the field as a host path, so a stage
    whose files promoted correctly still refuses with ``receipt published output
    is missing`` for a file the same call just wrote.

    Rewrite one of those fields only when promotion placed that exact
    run-root-relative destination and the bytes there already carry the sha256
    the record declares. A value that fails either test is left alone so the
    gate still refuses it. The provider's own spelling is kept alongside as
    ``provider_published_path`` and the manifest digest is recomputed, because
    the host is now the party attesting to where these files are.
    """
    changes: list[dict[str, str]] = []
    empty = {"changed": False, "changes": changes}
    if not isinstance(promotion, dict):
        return empty
    placed = sorted(
        {
            relative
            for key in ("promoted", "unchanged")
            for relative in (promotion.get(key) or [])
            if isinstance(relative, str) and relative
        }
    )
    manifest = receipt.get("output_manifest")
    if not placed or not isinstance(manifest, dict):
        return empty
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, list):
        return empty

    def rehome(holder: dict[str, Any], declared_digest: Any) -> None:
        value = holder.get("published_path")
        if not isinstance(value, str) or not value:
            return
        if Path(value).expanduser().is_file():
            return
        matches = [relative for relative in placed if value.endswith("/" + relative)]
        if len(matches) != 1:
            return
        destination = (run_root / matches[0]).resolve()
        if not destination.is_file():
            return
        if not isinstance(declared_digest, str) or sha256_file(destination) != declared_digest:
            return
        holder["published_path"] = str(destination)
        holder["provider_published_path"] = value
        changes.append({"from": value, "to": str(destination), "sha256": declared_digest})

    for artifact in artifacts:
        if not isinstance(artifact, dict):
            continue
        files = artifact.get("files")
        files = files if isinstance(files, list) else []
        for record in files:
            if isinstance(record, dict):
                rehome(record, record.get("sha256"))
        if len(files) == 1 and isinstance(files[0], dict):
            rehome(artifact, files[0].get("sha256"))
    if not changes:
        return empty
    provider_digest = manifest.get("output_manifest_sha256")
    manifest["output_manifest_sha256"] = sha256_json(artifacts)
    return {
        "changed": True,
        "changes": changes,
        "provider_output_manifest_sha256": provider_digest,
        "output_manifest_sha256": manifest["output_manifest_sha256"],
    }


def rehome_attempt_stage_receipt(
    *,
    run_root: Path,
    stage_id: str,
    provider_receipt: dict[str, Any],
    payload: bytes,
    attempt_id: Any,
) -> str | None:
    """Apply the same rehoming to the attempt-local copy of the stage receipt.

    ``validate_completed_receipt`` requires the canonical receipt and the
    attempt's own ``stage-receipt.json`` to parse equal, and the attempt copy is
    one of the files the provider returned, so it carries the same container
    paths. Correcting only the canonical copy trades one refusal for another.
    Rewrite the attempt copy only when it still holds exactly the receipt the
    provider sent, which keeps this idempotent and stops it from touching a file
    that is already something else.
    """
    if not isinstance(attempt_id, str) or not attempt_id:
        return None
    attempt_receipt = (
        run_root / "artifacts" / "stages" / stage_id / "attempts" / attempt_id / "stage-receipt.json"
    )
    if not attempt_receipt.is_file():
        return None
    try:
        existing = json.loads(attempt_receipt.read_bytes())
    except json.JSONDecodeError:
        return None
    if existing != provider_receipt:
        return None
    staging = attempt_receipt.with_name(f".{attempt_receipt.name}.{uuid.uuid4().hex}.incoming")
    staging.write_bytes(payload)
    os.replace(staging, attempt_receipt)
    return str(attempt_receipt)


def install_harvested_receipt(
    receipt_path: Path, *, run_root: Path, stage_id: str, promotion: Any = None
) -> dict[str, Any]:
    """Install one verified provider receipt where local downstream stages read it."""
    destination = run_root / "artifacts" / "receipts" / f"{stage_id}.json"
    payload = receipt_path.read_bytes()
    rehomed: dict[str, Any] | None = None
    if promotion is not None:
        provider_receipt = json.loads(payload)
        receipt = json.loads(payload)
        result = rehome_provider_published_paths(
            receipt, run_root=run_root, promotion=promotion
        )
        if result["changed"]:
            receipt["host_rehome"] = result
            payload = (json.dumps(receipt, indent=2, sort_keys=True) + "\n").encode("utf-8")
            result["attempt_receipt_path"] = rehome_attempt_stage_receipt(
                run_root=run_root,
                stage_id=stage_id,
                provider_receipt=provider_receipt,
                payload=payload,
                attempt_id=receipt.get("attempt_id"),
            )
            rehomed = result
    digest = hashlib.sha256(payload).hexdigest()
    if destination.is_file():
        existing = sha256_file(destination)
        if existing != digest:
            return {
                "ok": False,
                "path": str(destination),
                "errors": ["a different canonical receipt already exists for this stage"],
            }
        return {
            "ok": True,
            "path": str(destination),
            "sha256": digest,
            "unchanged": True,
            "rehomed": rehomed,
        }
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = destination.parent / f".{destination.name}.{digest[:12]}.incoming"
    staging.write_bytes(payload)
    os.replace(staging, destination)
    return {
        "ok": True,
        "path": str(destination),
        "sha256": digest,
        "unchanged": False,
        "rehomed": rehomed,
    }


def commit_harvested_stage(
    *,
    run_root: Path,
    bundle_root: Path,
    plan: dict[str, Any],
    stage_id: str,
) -> dict[str, Any]:
    """Commit one returned provider stage to the canonical resumable run state.

    Artifact promotion and receipt installation make downstream files readable,
    but they do not by themselves make ``execute --resume`` safe.  The executor
    resumes from its checkpoint, progress ledger, status, and run pointer.  Keep
    those four records in one host-side critical section after every successful
    harvest so a provider stage has the same durable completion semantics as a
    locally executed stage.
    """
    from claude_binder.lane import (
        reconcile_run_checkpoint,
        stage_event,
        update_run_pointer,
        verify_run_bundle,
    )

    run_root = Path(run_root).resolve()
    bundle_root = Path(bundle_root).resolve()
    plan_path = bundle_root / "run-plan.json"
    bundle_check = verify_run_bundle(plan_path)
    if not bundle_check.get("ok"):
        return {
            "ok": False,
            "errors": [
                f"run bundle verification failed: {error}"
                for error in bundle_check.get("errors", [])
            ],
        }
    on_disk_plan = load_json(plan_path)
    if on_disk_plan.get("run_fingerprint") != plan.get("run_fingerprint"):
        return {"ok": False, "errors": ["harvest plan does not match the run bundle"]}

    ordered = list(plan.get("ordered_stage_ids") or [])
    if not ordered:
        ordered = [
            item.get("stage_id")
            for item in plan.get("stages", [])
            if isinstance(item, dict) and isinstance(item.get("stage_id"), str)
        ]
    stage_map = {
        item.get("stage_id"): item
        for item in plan.get("stages", [])
        if isinstance(item, dict) and isinstance(item.get("stage_id"), str)
    }
    adapter_map = {
        item.get("adapter_id"): item
        for item in plan.get("adapters", [])
        if isinstance(item, dict) and isinstance(item.get("adapter_id"), str)
    }
    if stage_id not in ordered or stage_id not in stage_map:
        return {"ok": False, "errors": [f"harvested stage is absent from the plan: {stage_id}"]}

    receipts_dir = run_root / "artifacts" / "receipts"
    with dispatch_lock(run_root):
        checkpoint = reconcile_run_checkpoint(
            run_root,
            plan=plan,
            bundle_sha256=str(bundle_check["bundle_sha256"]),
            ordered_stage_ids=ordered,
            stage_map=stage_map,
            adapter_map=adapter_map,
        )
        completed = list(checkpoint.get("completed_stage_ids", []))
        if checkpoint.get("ok") is not True:
            return {
                "ok": False,
                "errors": list(checkpoint.get("errors", [])),
                "completed_stage_ids": completed,
            }
        if stage_id not in completed:
            missing_dependencies = [
                dependency
                for dependency in stage_map[stage_id].get("depends_on", [])
                if dependency not in completed
            ]
            reason = (
                "a dependency receipt is missing: " + ", ".join(missing_dependencies)
                if missing_dependencies
                else "its canonical receipt did not pass checkpoint reconciliation"
            )
            return {
                "ok": False,
                "errors": [f"provider stage {stage_id} was not checkpointed because {reason}"],
                "completed_stage_ids": completed,
            }
        progress_path = run_root / "artifacts" / "stage-progress.jsonl"
        progress_rows = load_jsonl(progress_path) if progress_path.is_file() else []
        progress_recorded_stage_ids: list[str] = []
        admitted_stage_ids = set(checkpoint.get("reconciled_stage_ids", [])) | {stage_id}
        for admitted_stage_id in ordered:
            if admitted_stage_id not in admitted_stage_ids or admitted_stage_id not in completed:
                continue
            admitted_receipt_path = receipts_dir / f"{admitted_stage_id}.json"
            receipt_digest = sha256_file(admitted_receipt_path)
            already_recorded = any(
                row.get("stage_id") == admitted_stage_id
                and row.get("status") == "completed"
                and row.get("receipt_sha256") == receipt_digest
                for row in progress_rows
            )
            if already_recorded:
                continue
            event = stage_event(
                admitted_stage_id,
                "completed",
                "guarded Modal harvest committed",
                {
                    "receipt_sha256": receipt_digest,
                    "completion_source": "guarded-modal-wave-dispatcher",
                },
            )
            append_jsonl(progress_path, event)
            progress_rows.append(event)
            progress_recorded_stage_ids.append(admitted_stage_id)

        status_path = run_root / "status.json"
        status = load_json(status_path) if status_path.is_file() else {}
        status.update(
            {
                "schema_version": 1,
                "run_id": plan.get("run_id"),
                "run_fingerprint": plan.get("run_fingerprint"),
                "state": "running",
                "ok": False,
                "completed_stages": completed,
                "updated_at": utc_now(),
            }
        )
        status.setdefault("started_at", utc_now())
        status.setdefault("resumed_stages", [])
        status.setdefault("skipped_stages", [])
        status.setdefault("skipped_stage_details", [])
        write_json(status_path, status)
        pointer = update_run_pointer(
            run_root,
            bundle_root=bundle_root,
            completed_stage_ids=completed,
            state="running",
        )
        checkpoint_updated_at = load_json(run_root / "stage-checkpoint.json").get("updated_at")

    return {
        "ok": True,
        "stage_id": stage_id,
        "completed_stage_ids": completed,
        "checkpoint": str(run_root / "stage-checkpoint.json"),
        "checkpoint_updated_at": checkpoint_updated_at,
        "progress_recorded": stage_id in progress_recorded_stage_ids,
        "progress_recorded_stage_ids": progress_recorded_stage_ids,
        "run_pointer": str(run_root / "run-pointer.json") if pointer is not None else None,
    }


def harvest_completed_receipt(
    workspace: Path,
    row: dict[str, Any],
    *,
    plan: dict[str, Any],
    stage: dict[str, Any],
    run_root: Path | None = None,
) -> dict[str, Any]:
    """Load and validate a completed receipt copied out of ``./out``.

    Pass ``run_root`` to also promote the job's returned artifact blobs into the
    host run root. Without it the receipt comes home and the structures it
    describes stay on the Volume, so the first local stage downstream reads an
    empty directory. The result then carries ``artifact_promotion`` as the string
    ``"not requested"``, which is the visible marker that the transport was
    skipped rather than run and found empty.
    """
    job_id = str(row["job_id"])
    receipt_path = harvested_receipt_path(workspace, job_id, row.get("output_files"))
    if not receipt_path.is_file():
        return {
            "ok": False,
            "stage_id": stage["stage_id"],
            "job_id": job_id,
            "errors": [f"harvested receipt is missing: {receipt_path}"],
        }
    try:
        receipt = load_json(receipt_path)
    except Exception as exc:
        return {
            "ok": False,
            "stage_id": stage["stage_id"],
            "job_id": job_id,
            "errors": [f"harvested receipt is unreadable: {type(exc).__name__}: {exc}"],
        }
    if receipt.get("stage_id") != stage["stage_id"]:
        return {
            "ok": False,
            "stage_id": stage["stage_id"],
            "job_id": job_id,
            "errors": ["harvested receipt belongs to another stage"],
        }
    if receipt.get("run_fingerprint") != plan.get("run_fingerprint"):
        return {
            "ok": False,
            "stage_id": stage["stage_id"],
            "job_id": job_id,
            "errors": ["harvested receipt belongs to another run"],
        }
    if receipt.get("attempt_id") != row.get("attempt_id"):
        return {
            "ok": False,
            "stage_id": stage["stage_id"],
            "job_id": job_id,
            "errors": ["harvested receipt belongs to another attempt"],
        }
    if run_root is not None:
        adapters = {
            item.get("adapter_id"): item
            for item in plan.get("adapters", [])
            if isinstance(item, dict)
        }
        adapter = adapters.get(stage.get("adapter_id"))
        if not isinstance(adapter, dict):
            return {
                "ok": False,
                "stage_id": stage["stage_id"],
                "job_id": job_id,
                "errors": ["cannot recompute harvested stage identity: adapter is missing"],
            }
        dependency_hashes = {}
        for dependency in stage.get("depends_on", []):
            dependency_path = run_root / "artifacts" / "receipts" / f"{dependency}.json"
            dependency_hashes[dependency] = (
                sha256_file(dependency_path) if dependency_path.is_file() else None
            )
        expected_stage_identity = sha256_json(
            {
                "run_fingerprint": plan["run_fingerprint"],
                "stage": stage,
                "adapter": adapter,
                "dependency_receipts": dependency_hashes,
            }
        )
        if not isinstance(receipt.get("stage_identity"), str) or receipt.get(
            "stage_identity"
        ) != expected_stage_identity:
            return {
                "ok": False,
                "stage_id": stage["stage_id"],
                "job_id": job_id,
                "errors": ["harvested receipt stage identity does not match host dependencies"],
            }
    validation_path = harvested_validation_path(
        workspace, job_id, row.get("output_files")
    )
    if validation_path.is_file():
        try:
            validation = load_json(validation_path)
        except Exception as exc:
            return {
                "ok": False,
                "stage_id": stage["stage_id"],
                "job_id": job_id,
                "errors": [
                    f"harvested receipt validation is unreadable: {type(exc).__name__}: {exc}"
                ],
            }
    else:
        validation = None
    check = validate_harvested_receipt(
        receipt,
        validation,
        receipt_sha256=sha256_file(receipt_path),
    )
    if run_root is None:
        promotion: Any = "not requested"
    else:
        promotion = promote_returned_artifacts(
            workspace,
            row,
            run_root=run_root,
            receipt=receipt,
            receipt_sha256=sha256_file(receipt_path),
            stage=stage,
        )
    result = {
        **check,
        "job_id": job_id,
        "receipt_path": str(receipt_path),
        "validation_path": str(validation_path),
        "receipt": receipt,
        "artifact_promotion": promotion,
        "receipt_install": "not requested",
        "campaign_commit": "not requested",
    }
    if isinstance(promotion, dict) and not promotion["ok"]:
        # A receipt that validates while its artifacts never landed is the exact
        # shape that let a Modal stage look finished to the dispatcher and empty
        # to the next local stage. Fail the harvest instead.
        result["ok"] = False
        result["errors"] = list(result.get("errors") or []) + [
            f"artifact promotion failed: {error}" for error in promotion["errors"]
        ]
    if result["ok"] and run_root is not None:
        installed = install_harvested_receipt(
            receipt_path,
            run_root=run_root,
            stage_id=stage["stage_id"],
            promotion=promotion,
        )
        result["receipt_install"] = installed
        if not installed["ok"]:
            result["ok"] = False
            result["errors"] = list(result.get("errors") or []) + [
                f"receipt install failed: {error}" for error in installed["errors"]
            ]
    if result["ok"] and run_root is not None:
        runtime = plan.get("runtime")
        bundle_value = runtime.get("bundle_path") if isinstance(runtime, dict) else None
        if not isinstance(bundle_value, str) or not bundle_value:
            result["ok"] = False
            result["errors"] = list(result.get("errors") or []) + [
                "campaign commit failed: plan.runtime.bundle_path is missing"
            ]
        else:
            committed = commit_harvested_stage(
                run_root=run_root,
                bundle_root=Path(bundle_value),
                plan=plan,
                stage_id=stage["stage_id"],
            )
            result["campaign_commit"] = committed
            if not committed["ok"]:
                result["ok"] = False
                result["errors"] = list(result.get("errors") or []) + [
                    f"campaign commit failed: {error}" for error in committed["errors"]
                ]
    return result


def merge_register(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Join the register's three row kinds into one row per job.

    The register is append-only, and a job writes several rows at different
    times. ``submit_wave`` writes the submission, which knows the stage, the
    phase and the shard but not the outcome. ``collect_notifications`` writes
    the completion, which knows the outcome but carries only the job id.
    ``close_wave`` writes the close, which says whether the handle was closed
    and therefore whether the Volume commit is guaranteed. Reading one kind
    alone answers nothing, so every reader goes through this.

    A completion, terminal result and close can each arrive more than once, so
    the latest non-null value within a row kind wins. Row-kind semantics then
    determine precedence: terminal evidence overrides reconstructed completion
    evidence regardless of replay order, and close metadata is applied last.

    The result keeps submission order. A job with no completion or terminal
    result keeps ``state == "submitted"``, which marks it as possibly running.
    """
    buckets: dict[str, dict[str, dict[str, Any]]] = {}
    order: list[str] = []
    for row in rows:
        job_id = row.get("job_id")
        if not job_id:
            continue
        if job_id not in buckets:
            buckets[job_id] = {
                "submission": {},
                "completion": {},
                "terminal": {},
                "close": {},
            }
            order.append(job_id)
        record = row.get("record")
        kind = record if record in {"completion", "terminal", "close"} else "submission"
        bucket = buckets[job_id][kind]
        if kind == "submission":
            for key, value in row.items():
                bucket.setdefault(key, value)
        else:
            bucket.update({key: value for key, value in row.items() if value is not None})
    merged: dict[str, dict[str, Any]] = {}
    for job_id in order:
        combined: dict[str, Any] = {}
        # Row semantics determine precedence. This keeps an authoritative
        # terminal result authoritative even if a replay presents the
        # append-only register in another order.
        for kind in ("submission", "completion", "terminal", "close"):
            combined.update(buckets[job_id][kind])
        terminal_state = buckets[job_id]["terminal"].get("terminal_state")
        if terminal_state is not None:
            combined["state"] = terminal_state
        combined.setdefault("state", "submitted")
        merged[job_id] = combined
    return [merged[job_id] for job_id in order]


def find_stage_receipt(
    workspace: Path,
    stage_id: str,
    register: Sequence[dict[str, Any]],
    *,
    receipts_dir: Path | None = None,
) -> dict[str, Any] | None:
    """Read a completed stage's receipt from the host's own disk.

    The canonical receipt lives on the Volume, which an ordinary process cannot
    read. The finalize job copies it into ``./out/``, so the harvest leaves a
    readable copy at ``hpc/<job_id>/receipt.json``. This walks the job register
    backwards to the newest succeeded finalize job for the stage and reads that.

    ``register`` is the raw rows. The join happens here, so callers never have to
    remember that a submission and a completion are two different rows.

    A stage that ran on the host leaves no job to harvest, so none of that applies to
    it. ``receipts_dir`` names the run root's own receipts directory, where the
    executor writes ``<stage_id>.json`` for every stage it runs itself. It is read
    only when the register yields nothing, so a Modal stage still reads the
    provider's own copy, and the local receipt is accepted only if it names this
    stage and reports success.

    Both branches require the receipt to report ``ok`` true, and the harvested
    branch also requires the job's own exit code to be zero. Two callers depend
    on it: ``resolve_scale_count`` sizes the next paid wave from the returned
    manifest, and ``resume_report`` tells an operator which stages are done.
    """
    for row in reversed(merge_register(register)):
        if row.get("stage_id") != stage_id:
            continue
        if row.get("phase") not in {"finalize", "single"}:
            continue
        # State and exit code are two conditions, not one. Every other gate in
        # this file reads them that way, and ``modal_platform.complete_stage``
        # refuses a succeeded job whose container exited non-zero outright. This
        # branch used to read the state alone and then return whatever JSON sat
        # at the path, which is how a receipt saying ``ok: false`` became the
        # answer ``resume_report`` gives an operator asking what is left to run.
        if row.get("state") != "succeeded" or row.get("exit_code") not in (None, 0):
            continue
        path = harvested_receipt_path(
            workspace, str(row["job_id"]), row.get("output_files")
        )
        if not path.is_file():
            continue
        receipt = load_json(path)
        # The local branch below has always required this. A harvested receipt
        # is not more trustworthy than a local one for having crossed a network.
        if isinstance(receipt, dict) and receipt.get("ok") is True:
            return receipt
    if receipts_dir is not None:
        local_path = receipts_dir / f"{stage_id}.json"
        if local_path.is_file():
            receipt = load_json(local_path)
            manifest = receipt.get("output_manifest") or {}
            if (
                receipt.get("stage_id") == stage_id
                and receipt.get("ok") is True
                and manifest.get("stage_id") == stage_id
                and manifest.get("artifacts")
            ):
                return receipt
    return None


def plan_receipts_dir(plan: dict[str, Any]) -> Path | None:
    """Return the run root's receipts directory as the plan itself records it.

    ``runtime.bundle_path`` is the only run-root path a materialized plan carries, and
    the receipts sit beside the bundle under ``artifacts/receipts``. A resumed run in a
    fresh workspace makes that path stale, which is why every caller can override it.
    """
    bundle = (plan.get("runtime") or {}).get("bundle_path")
    if not isinstance(bundle, str) or not bundle:
        return None
    return Path(bundle).parent / "artifacts" / "receipts"


def resolve_scale_count(
    stage: dict[str, Any],
    plan: dict[str, Any],
    *,
    workspace: Path,
    register: Sequence[dict[str, Any]],
    receipts_dir: Path | None = None,
) -> dict[str, Any]:
    """Count the upstream artifact this stage fans out over.

    This is the reason the dispatcher cannot plan the whole run up front. A
    fan-out stage declares ``fanout.count_from`` naming an upstream stage and
    one of its artifacts. The real width exists only after that stage finishes.
    The estimator's numbers in the plan are upper bounds and this checks against
    them, exactly as the executor's own stage_scale_count does.

    A resumed run calls this again rather than trusting a recorded number,
    because a rerun upstream stage can legitimately change the width.
    """
    fanout = stage.get("fanout") or {}
    if fanout.get("scale_count") is not None:
        width = int(fanout["scale_count"])
        return {
            "stage_id": stage["stage_id"],
            "resolved_width": width,
            "ceiling": validate_scale_count(stage, plan, width),
            "source": "fanout.scale_count",
        }

    source = fanout["count_from"]
    # The upstream stage may have run here rather than on Modal, in which case the job
    # register holds nothing to harvest and the only receipt is the local one.
    local_receipts = receipts_dir if receipts_dir is not None else plan_receipts_dir(plan)
    receipt = find_stage_receipt(
        workspace, source["stage_id"], register, receipts_dir=local_receipts
    )
    if receipt is None:
        tried = f"the Modal job register, and {local_receipts}" if local_receipts else "the Modal job register"
        raise LookupError(
            f"stage {stage['stage_id']} fans out over {source['stage_id']}:"
            f"{source['artifact_id']}, and no usable receipt for that stage was found; tried {tried}"
        )
    artifacts = receipt["output_manifest"]["artifacts"]
    matching = [item for item in artifacts if item["artifact_id"] == source["artifact_id"]]
    if not matching:
        raise LookupError(
            f"receipt for {source['stage_id']} has no artifact {source['artifact_id']!r}"
        )
    # A smoke_scale stage writes one artifact row per phase. The scale row is the
    # one that carries the full width; the smoke row always reads one.
    artifact = next(
        (item for item in matching if item.get("phase") in {"scale", "single"}), matching[-1]
    )
    if source.get("value") == "records":
        width = sum(int(item.get("records", 0)) for item in artifact["files"])
    else:
        width = int(artifact["count"])

    ceiling = validate_scale_count(stage, plan, width)

    return {
        "stage_id": stage["stage_id"],
        "resolved_width": width,
        "ceiling": ceiling,
        "source": "fanout.count_from",
        "source_stage_id": source["stage_id"],
        "source_artifact_id": source["artifact_id"],
        "source_value": source.get("value", "count"),
        "source_receipt_sha256": sha256_json(receipt),
    }


def validate_scale_count(stage: dict[str, Any], plan: dict[str, Any], width: int) -> int:
    """Apply the lane's positive and runtime-ceiling checks to any scale source."""
    if width < 1:
        raise ValueError(f"stage {stage['stage_id']} resolved a nonpositive scale count")
    ceiling = fanout_ceiling(stage, plan)
    if width > ceiling:
        raise ValueError(
            f"stage {stage['stage_id']} resolved {width} candidates against a runtime cap of {ceiling}"
        )
    return ceiling


def fanout_ceiling(stage: dict[str, Any], plan: dict[str, Any]) -> int:
    """The runtime cap this stage's width is checked against.

    These three branches mirror the executor's stage_scale_count so that a width
    this module accepts is one the lane would also accept.
    """
    estimate = plan.get("fanout_estimate", {}).get("counts", {})
    runtime_cap = int(plan["runtime"]["maximum_generated_candidates"])
    stage_id = stage["stage_id"]
    if stage_id.startswith("cofold-rescore-"):
        return int(estimate.get("rescore_candidates_upper_bound", runtime_cap))
    if stage_id.startswith("optimization-cofold-round-"):
        return int(estimate.get("optimized_variants_upper_bound", runtime_cap))
    return runtime_cap


def scale_wave_width(plan: dict[str, Any]) -> int:
    """Jobs per scale subwave, from the campaign or from the default.

    The width follows what the user has configured on their Modal provider row,
    which the campaign carries in `provider.maximum_concurrent_jobs`. That
    number is theirs to set and this module never reads it from the platform,
    because a value this module cannot verify is a worse input than a stated
    one. A campaign that omits it gets DEFAULT_SCALE_WAVE_WIDTH, which is the
    platform's own default for a row nobody has changed.

    A width outside the provider's reviewed 1..10,000 setting range is refused;
    it is never silently floored or clamped.
    """
    provider = plan.get("provider")
    stated = provider.get("maximum_concurrent_jobs") if isinstance(provider, dict) else None
    if stated is None:
        return DEFAULT_SCALE_WAVE_WIDTH
    return bounded_integer(
        stated,
        label="provider.maximum_concurrent_jobs",
        minimum=1,
        maximum=MAXIMUM_PROVIDER_CONCURRENT_JOBS,
    )


# --- planning one stage ------------------------------------------------------


def plan_stage(
    plan: dict[str, Any],
    stage: dict[str, Any],
    *,
    policy: DispatchPolicy,
    workspace: Path,
    register: Sequence[dict[str, Any]],
    attempt_id: str | None = None,
    receipts_dir: Path | None = None,
) -> dict[str, Any]:
    """Return the waves of jobs one stage needs, in submission order.

    A wave is a set of jobs that can be submitted in the same turn. Waves are
    strictly ordered, and the dispatcher must collect a wave before it submits
    the next one.

    A ``single`` stage is one wave of one job. A ``smoke_scale`` stage has a
    smoke wave, scale subwaves each `scale_wave_width` wide, and a finalize
    wave. The smoke wave
    exists as its own wave because its whole purpose is to fail before the
    expensive shards are created. Each scale subwave is closed before the next
    one is submitted, so its Volume writes are committed before the finalize
    job can read them.
    """
    adapter_map = {item["adapter_id"]: item for item in plan["adapters"]}
    adapter = adapter_map[stage["adapter_id"]]
    attempt_id = attempt_id or uuid.uuid4().hex

    if stage["mode"] == "single":
        spec = build_job_spec(
            plan,
            stage,
            adapter,
            attempt_id=attempt_id,
            phase="single",
            count=1,
            shard=None,
            policy=policy,
            finalize=True,
            scale_count=1,
        )
        return {
            "stage_id": stage["stage_id"],
            "attempt_id": attempt_id,
            "mode": "single",
            "waves": [[spec]],
            "fanout": None,
        }

    resolved = resolve_scale_count(
        stage, plan, workspace=workspace, register=register, receipts_dir=receipts_dir
    )
    width = int(resolved["resolved_width"])
    shards = split_shards(
        width,
        policy.width_for(adapter["adapter_id"], ceiling=int(resolved["ceiling"])),
    )
    resolved["shard_count"] = len(shards)
    resolved["attempt_id"] = attempt_id
    resolved["resolved_at"] = utc_now()

    smoke = build_job_spec(
        plan, stage, adapter,
        attempt_id=attempt_id, phase="smoke", count=1, shard=None,
        policy=policy, finalize=False, run_parser=True, scale_count=width,
    )
    scale = [
        build_job_spec(
            plan, stage, adapter,
            attempt_id=attempt_id, phase="scale", count=width, shard=shard,
            policy=policy, finalize=False, run_parser=False, scale_count=width,
        )
        for shard in shards
    ]
    final = build_job_spec(
        plan, stage, adapter,
        attempt_id=attempt_id, phase="finalize", count=width, shard=None,
        policy=policy, finalize=True, merge_phase="scale", run_parser=True,
        scale_count=width,
        expected_shards=[item.as_dict() for item in shards],
    )
    wave_width = scale_wave_width(plan)
    scale_waves = [
        scale[index : index + wave_width]
        for index in range(0, len(scale), wave_width)
    ]
    return {
        "stage_id": stage["stage_id"],
        "attempt_id": attempt_id,
        "mode": "smoke_scale",
        "waves": [[smoke], *scale_waves, [final]],
        "fanout": resolved,
    }


def stage_waves(plan: dict[str, Any]) -> list[list[str]]:
    """Group stages into dependency waves.

    A wave holds every stage whose ``depends_on`` edges are already satisfied by
    earlier waves. Wave depth, not call count, sets the number of agent round
    trips a run costs.
    """
    stages = {stage["stage_id"]: stage for stage in plan["stages"]}
    pending = {
        stage_id: set(stage.get("depends_on", [])) for stage_id, stage in stages.items()
    }
    done: set[str] = set()
    waves: list[list[str]] = []
    while pending:
        ready = sorted(
            stage_id for stage_id, deps in pending.items() if deps <= done
        )
        if not ready:
            raise ValueError(
                f"stage dependency cycle among: {', '.join(sorted(pending))}"
            )
        waves.append(ready)
        done.update(ready)
        for stage_id in ready:
            pending.pop(stage_id)
    return waves


# --- dispatch ----------------------------------------------------------------
#
# Everything below needs the `host` object. It exists only inside the Claude
# Science `repl` kernel, where it is pre-bound rather than importable, so every
# function here takes it as an argument. None of this runs from the command
# line, and the command line does not pretend it can.


def stage_inputs(
    spec: JobSpec, *, workspace: Path, policy: DispatchPolicy
) -> list[dict[str, str]]:
    """Write a job's input files into the workspace and return the input list.

    ``inputs=`` entries name a workspace-relative ``src`` and a bare filename
    ``dst``. They stage flat into the container's working directory root, so a
    ``dst`` carrying a directory is refused at submit. Anything needing a
    directory layout has to be made by the command itself.

    The package source is not uploaded with each job. It lives on the Volume at
    ``policy.repo_dir``, because a Volume costs zero upload per submit and this
    run submits roughly sixty jobs. See ``repo_bootstrap_note``.
    """
    shard_part = "single" if spec.shard is None else f"{spec.shard.index:04d}"
    staging = (
        workspace
        / policy.workspace_staging
        / spec.attempt_id
        / spec.stage_id
        / spec.phase
        / shard_part
    )
    staging.mkdir(parents=True, exist_ok=True)

    if spec.phase == "bootstrap":
        package_destination = spec.bootstrap_package_destination
        if not isinstance(package_destination, str) or not package_destination:
            raise ValueError(
                "bootstrap job has no package destination; rebuild the bootstrap spec before submit"
            )
        script_path = staging / BOOTSTRAP_SCRIPT
        script_path.write_text(spec.script_text)
        archive_path = staging / BOOTSTRAP_ARCHIVE
        with tarfile.open(archive_path, "w:gz") as archive:
            for item in spec.bootstrap_files:
                source = Path(item["source"])
                if not source.is_file():
                    raise FileNotFoundError(
                        f"bootstrap input is missing at submit: {source}"
                    )
                archive.add(source, arcname=str(item["destination"]), recursive=False)
        manifest_path = staging / BOOTSTRAP_MANIFEST
        write_json(
            manifest_path,
            {
                "schema_version": 1,
                "volume_mount": policy.volume_mount,
                "package_destination": package_destination,
                "files": spec.bootstrap_files,
            },
        )
        inputs = [
            {"src": str(script_path.relative_to(workspace)), "dst": BOOTSTRAP_SCRIPT},
            {"src": str(archive_path.relative_to(workspace)), "dst": BOOTSTRAP_ARCHIVE},
            {"src": str(manifest_path.relative_to(workspace)), "dst": BOOTSTRAP_MANIFEST},
        ]
        if len(inputs) > INPUT_FILE_CAP:
            raise ValueError(
                f"a submit may carry at most {INPUT_FILE_CAP} input files; this one has {len(inputs)}"
            )
        total_bytes = sum((workspace / item["src"]).stat().st_size for item in inputs)
        if total_bytes > INPUT_BYTES_CAP:
            raise ValueError(
                f"inputs total {total_bytes} bytes and the per-submit cap is {INPUT_BYTES_CAP}"
            )
        return inputs

    script_path = staging / "shard.sh"
    script_path.write_text(spec.script_text)
    inputs = [{"src": str(script_path.relative_to(workspace)), "dst": "shard.sh"}]

    receipt_path = staging / RECEIPT_SCRIPT
    receipt_path.write_text(spec.receipt_script_text)
    inputs.append(
        {"src": str(receipt_path.relative_to(workspace)), "dst": RECEIPT_SCRIPT}
    )

    if spec.shard is not None:
        manifest_path = staging / "shard-manifest.json"
        write_json(
            manifest_path,
            {
                "stage_id": spec.stage_id,
                "adapter_id": spec.adapter_id,
                "attempt_id": spec.attempt_id,
                "phase": spec.phase,
                **spec.shard.as_dict(),
            },
        )
        inputs.append(
            {"src": str(manifest_path.relative_to(workspace)), "dst": "shard-manifest.json"}
        )

    if spec.phase == "finalize" and spec.expected_shards:
        expected_path = staging / "expected-shards.json"
        write_json(expected_path, {"shards": spec.expected_shards})
        inputs.append(
            {
                "src": str(expected_path.relative_to(workspace)),
                "dst": "expected-shards.json",
            }
        )

    total_bytes = sum((workspace / item["src"]).stat().st_size for item in inputs)
    if len(inputs) > INPUT_FILE_CAP:
        raise ValueError(
            f"a submit may carry at most {INPUT_FILE_CAP} input files; this one has {len(inputs)}"
        )
    if total_bytes > INPUT_BYTES_CAP:
        raise ValueError(
            f"inputs total {total_bytes} bytes and the per-submit cap is {INPUT_BYTES_CAP}"
        )
    return inputs


def _volume_relative_path(path: str, mount: str) -> str:
    """Map an absolute container path to an archive member under its Volume."""
    path_text = str(path).rstrip("/")
    mount_text = str(mount).rstrip("/")
    if path_text != mount_text and not path_text.startswith(mount_text + "/"):
        raise ValueError(
            f"container path {path_text!r} is outside the configured Volume mount {mount_text!r}"
        )
    relative = path_text[len(mount_text) :].lstrip("/")
    if not relative:
        raise ValueError(f"container path {path_text!r} points at the Volume root")
    return relative


def modal_producer_stages(register: Sequence[dict[str, Any]]) -> set[str]:
    """Return the stages the job register proves ran to success on the provider.

    Two row kinds have to be joined. A ``submitted`` row carries the stage id and
    the provider job id; the completion row carries the outcome and the job id but
    no stage id. A bootstrap job populates the Volume and writes no artifact, so
    its success is not evidence that the stage produced anything.

    This is the only admissible reason to dispatch a paid stage whose declared
    input was not returned to the host: the upstream stage already wrote it to
    the same Volume.
    """
    return {
        str(row["stage_id"])
        for row in merge_register(register)
        if row.get("phase") in {"single", "finalize"}
        and row.get("state") == "succeeded"
        and row.get("exit_code") == 0
        and isinstance(row.get("stage_id"), str)
    }


def receipt_file_digests(receipts_dir: Path | None) -> dict[str, dict[str, Any]]:
    """Index every file any host receipt declares, by the absolute path it records.

    A receipt is the only host-side statement of what a stage wrote and what those
    bytes hash to. A manifest row names a file another stage produced, and that
    producer's receipt declares the same path with a sha256 over the file. Looking
    the path up here keeps the transport declarative: it reads a receipt, it does
    not parse the path to guess which stage made it.
    """
    index: dict[str, dict[str, Any]] = {}
    if receipts_dir is None or not receipts_dir.is_dir():
        return index
    for path in sorted(receipts_dir.glob("*.json")):
        try:
            receipt = load_json(path)
        except (json.JSONDecodeError, OSError):
            continue
        if not isinstance(receipt, dict):
            continue
        for artifact in (receipt.get("output_manifest") or {}).get("artifacts") or []:
            if not isinstance(artifact, dict):
                continue
            for record in artifact.get("files") or []:
                if not isinstance(record, dict):
                    continue
                digest = record.get("sha256")
                if not isinstance(digest, str) or not digest:
                    continue
                size = record.get("bytes")
                for key in ("path", "published_path"):
                    value = record.get(key)
                    if isinstance(value, str) and value:
                        index[value] = {
                            "sha256": digest,
                            "bytes": size if isinstance(size, int) else None,
                            "receipt": path.stem,
                        }
    return index


def manifest_reference_files(
    manifest_path: Path,
    artifact_type: str,
    *,
    origin: str,
    policy: DispatchPolicy,
    receipt_digests: dict[str, dict[str, Any]] | None = None,
) -> tuple[list[dict[str, Any]], list[str]]:
    """Resolve the files a manifest artifact's rows point at.

    ``MANIFEST_REFERENCE_FIELDS`` names the reference-bearing fields for the
    artifact type and pairs each with the row field holding that file's sha256.
    An artifact type the registry does not name resolves to nothing.

    Every reference is shipped to the exact path the row stores. Rewriting a row
    to a canonical ``published_path`` would break the reference, because the row
    names the producing attempt's path.

    What verifies a reference depends on what its digest covers. A ``file-bytes``
    digest checks the transfer directly. A ``sequence-text`` digest covers the
    residue string rather than the FASTA, so the producing stage's receipt supplies
    the file digest instead, and the row digest is left to the adapter that reads
    the sequence. A reference with neither is shipped and reported unverified.
    """
    fields = MANIFEST_REFERENCE_FIELDS.get(artifact_type)
    if not fields:
        return [], []
    if receipt_digests is None:
        receipt_digests = {}
    resolved: list[dict[str, Any]] = []
    errors: list[str] = []
    for number, line in enumerate(manifest_path.read_text().splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            errors.append(f"{origin}: row {number} is not JSON: {error}")
            continue
        if not isinstance(row, dict):
            errors.append(f"{origin}: row {number} is not an object")
            continue
        for path_field, digest_field, digest_covers in fields:
            value = row.get(path_field)
            if value is None:
                continue
            label = f"{origin} row {number} {path_field}"
            if not isinstance(value, str) or not value:
                errors.append(f"{label}: is not a path")
                continue
            try:
                destination = _volume_relative_path(value, policy.volume_mount)
            except ValueError:
                errors.append(f"{label}: is outside the Volume mount: {value}")
                continue
            source = Path(value)
            if not source.is_file():
                errors.append(f"{label}: is missing on the host: {value}")
                continue
            row_digest = row.get(digest_field)
            row_digest = row_digest if isinstance(row_digest, str) and row_digest else None
            declared = receipt_digests.get(value)
            expected = declared["sha256"] if declared else None
            expected_bytes = declared["bytes"] if declared else None
            if digest_covers == FILE_BYTES:
                if expected is None:
                    expected = row_digest
                elif row_digest is not None and row_digest != expected:
                    errors.append(
                        f"{label}: the row declares {row_digest} and receipt "
                        f"{declared['receipt']} declares {expected}"
                    )
                    continue
            resolved.append(
                {
                    "source": str(source),
                    "destination": destination,
                    "sha256": expected,
                    "bytes": expected_bytes,
                    "origin": label,
                    "digest_source": (
                        f"receipt:{declared['receipt']}"
                        if declared
                        else (digest_field if expected else None)
                    ),
                }
            )
    return resolved, errors


def stage_input_files(
    plan: dict[str, Any],
    input_refs: Sequence[str],
    *,
    policy: DispatchPolicy,
    receipts_dir: Path | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, str]], list[str]]:
    """Resolve declared ``stage-id:artifact-id`` inputs to host files to ship.

    Returns the files to add to the bootstrap archive, the references that could
    not be resolved on the host, and the references that are wrong. The three are
    different things. An unresolved reference may be legitimate, because the
    upstream stage may have run on Modal and already written to the Volume; the
    caller decides that against the job register. A hash that disagrees with the
    receipt is never legitimate.

    The upstream receipt is the source of truth for where an artifact landed,
    because it records the published path the publisher actually wrote, and it is
    also the source of truth for the bytes, because it records a sha256 and a byte
    count for every declared file.

    An artifact whose type appears in ``MANIFEST_REFERENCE_FIELDS`` is a manifest
    of pointers, so its rows are followed and the files they name are shipped too.
    """
    if receipts_dir is None:
        receipts_dir = plan_receipts_dir(plan)
    resolved: list[dict[str, Any]] = []
    skipped: list[dict[str, str]] = []
    errors: list[str] = []
    seen: dict[str, dict[str, Any]] = {}
    receipt_digests = receipt_file_digests(receipts_dir)

    def keep(record: dict[str, Any]) -> None:
        """Add one file, refusing two different digests for one destination."""
        destination = record["destination"]
        first = seen.get(destination)
        if first is None:
            seen[destination] = record
            resolved.append(record)
            return
        digests = {first.get("sha256"), record.get("sha256")} - {None}
        if len(digests) > 1:
            errors.append(
                f"{record['origin']}: {destination} is already shipped from "
                f"{first['origin']} with a different sha256"
            )

    for input_ref in input_refs:
        source_stage, artifact_id = input_ref.split(":", 1)
        receipt_path = None if receipts_dir is None else receipts_dir / f"{source_stage}.json"
        if receipt_path is None or not receipt_path.is_file():
            skipped.append(
                {"input_ref": input_ref, "reason": f"no host receipt for {source_stage}"}
            )
            continue
        receipt = load_json(receipt_path)
        artifacts = [
            item
            for item in (receipt.get("output_manifest") or {}).get("artifacts") or []
            if item.get("artifact_id") == artifact_id
        ]
        if not artifacts:
            skipped.append(
                {"input_ref": input_ref, "reason": "the receipt declares no such artifact"}
            )
            continue
        found = 0
        for artifact in artifacts:
            artifact_type = artifact.get("artifact_type")
            for record in artifact.get("files") or []:
                for key in ("published_path", "path"):
                    value = record.get(key)
                    if not isinstance(value, str) or not value:
                        continue
                    source = Path(value)
                    if not source.is_file():
                        continue
                    try:
                        destination = _volume_relative_path(value, policy.volume_mount)
                    except ValueError:
                        continue
                    found += 1
                    declared_bytes = record.get("bytes")
                    declared_digest = record.get("sha256")
                    keep(
                        {
                            "source": str(source),
                            "destination": destination,
                            "sha256": declared_digest
                            if isinstance(declared_digest, str) and declared_digest
                            else None,
                            "bytes": declared_bytes if isinstance(declared_bytes, int) else None,
                            "origin": f"{input_ref} {key}",
                            "digest_source": f"receipt:{source_stage}",
                        }
                    )
                    if not isinstance(artifact_type, str):
                        continue
                    referenced, reference_errors = manifest_reference_files(
                        source,
                        artifact_type,
                        origin=input_ref,
                        policy=policy,
                        receipt_digests=receipt_digests,
                    )
                    errors.extend(reference_errors)
                    for item in referenced:
                        keep(item)
        if not found:
            skipped.append(
                {"input_ref": input_ref, "reason": "no readable host file under the Volume mount"}
            )
    return resolved, skipped, errors


def bootstrap_file_manifest(
    plan: dict[str, Any],
    stage: dict[str, Any],
    adapter: dict[str, Any],
    *,
    bundle_root: Path,
    source_repo: Path,
    policy: DispatchPolicy,
    receipts_dir: Path | None = None,
    register: Sequence[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Enumerate the run bundle, package subset and declared inputs for one stage.

    The stage contract proves that the selected stage belongs to this bundle.
    The plan's stage inputs name the upstream artifacts that must already exist
    on the same Volume. The bundle and package files are copied once, then the
    stage jobs read the committed Volume state.
    """
    contract_path = bundle_root / "stage-contract.json"
    if not contract_path.is_file():
        raise FileNotFoundError(f"stage contract is missing at bootstrap: {contract_path}")
    contract = load_json(contract_path)
    contract_stages = {
        item.get("stage_id"): item
        for item in contract.get("stages", [])
        if isinstance(item, dict)
    }
    if stage["stage_id"] not in contract_stages:
        raise ValueError(
            f"stage contract {contract_path} has no stage {stage['stage_id']!r}"
        )

    plan_stages = {item.get("stage_id"): item for item in plan.get("stages", [])}
    special_inputs = {"run-bundle", "run-bundle:controls", "stage-receipts"}
    dependencies = set(stage.get("depends_on", []))
    resolved_inputs: list[str] = []
    for input_ref in stage.get("inputs", []):
        if input_ref in special_inputs:
            continue
        resolved_inputs.append(input_ref)
        if not isinstance(input_ref, str) or ":" not in input_ref:
            raise ValueError(
                f"stage {stage['stage_id']} input is not a stage-id:artifact-id reference: {input_ref!r}"
            )
        source_stage, artifact_id = input_ref.split(":", 1)
        source = plan_stages.get(source_stage)
        if source is None:
            raise ValueError(
                f"stage {stage['stage_id']} input references unknown stage: {source_stage}"
            )
        if source_stage not in dependencies:
            raise ValueError(
                f"stage {stage['stage_id']} input source is not a direct dependency: {source_stage}"
            )
        if artifact_id not in {
            output.get("artifact_id")
            for output in source.get("outputs", [])
            if isinstance(output, dict)
        }:
            raise ValueError(
                f"stage {stage['stage_id']} input references unknown artifact: {input_ref}"
            )

    bundle_path = str(plan["runtime"]["bundle_path"])
    repo_path = str(policy.repo_dir)
    bundle_prefix = _volume_relative_path(bundle_path, policy.volume_mount)
    repo_prefix = _volume_relative_path(repo_path, policy.volume_mount)
    files: list[dict[str, Any]] = []

    def add_file(
        source: Path,
        destination: str,
        kind: str,
        *,
        expected_sha256: str | None = None,
        expected_bytes: int | None = None,
        origin: str | None = None,
    ) -> None:
        """Record one file for the archive, verifying it when something declared it.

        The bundle and the package are enumerated from disk, so nothing declares
        them and there is nothing to check them against. A stage input is
        different: the upstream receipt already recorded a sha256 and a byte count
        for it, and the container verifies the same digest after extraction. Not
        comparing here let changed bytes ship under a hash derived from the changed
        bytes, which the container then confirmed.
        """
        source = source.resolve()
        if not source.is_file():
            raise FileNotFoundError(f"bootstrap input is missing at plan time: {source}")
        if destination.startswith("/") or ".." in Path(destination).parts:
            raise ValueError(f"bootstrap destination is unsafe: {destination}")
        size = source.stat().st_size
        digest = hashlib.sha256(source.read_bytes()).hexdigest()
        label = origin or destination
        if expected_bytes is not None and expected_bytes != size:
            raise ValueError(
                f"{label}: {source} is {size} bytes and the receipt declares {expected_bytes}"
            )
        if expected_sha256 is not None and expected_sha256 != digest:
            raise ValueError(
                f"{label}: {source} hashes to {digest} and the receipt declares {expected_sha256}"
            )
        files.append(
            {
                "source": str(source),
                "destination": destination,
                "bytes": size,
                "sha256": digest,
                "kind": kind,
            }
        )

    bundle_files = sorted(
        path for path in bundle_root.rglob("*") if path.is_file() and not path.is_symlink()
    )
    if not bundle_files:
        raise ValueError(f"run bundle has no files: {bundle_root}")
    for path in bundle_files:
        add_file(path, posix_join(bundle_prefix, str(path.relative_to(bundle_root))), "run-bundle")

    package_root = source_repo / "src" / "claude_binder"
    if not package_root.is_dir():
        raise FileNotFoundError(f"package source is missing at bootstrap: {package_root}")
    package_files = sorted(
        path
        for path in package_root.rglob("*")
        if path.is_file()
        and not path.is_symlink()
        and path.suffix != ".pyc"
        and not PACKAGE_EXCLUDED_PARTS & set(path.relative_to(package_root).parts)
    )
    if not package_files:
        raise ValueError(f"package subset is empty: {package_root}")
    identity_path = bundle_root / "run-identity.json"
    if not identity_path.is_file():
        raise ValueError(
            "run bundle has no run-identity.json; rematerialize before a paid bootstrap"
        )
    identity = load_json(identity_path)
    identity_inputs = identity.get("identity_inputs")
    expected_package = (
        identity_inputs.get("package_file_sha256")
        if isinstance(identity_inputs, dict)
        else None
    )
    if not isinstance(expected_package, dict) or not expected_package:
        raise ValueError(
            "run identity has no package_file_sha256 manifest; rematerialize before a paid bootstrap"
        )
    actual_package = {
        path.relative_to(package_root).as_posix(): sha256_file(path)
        for path in package_files
    }
    if actual_package != expected_package:
        missing = sorted(expected_package.keys() - actual_package.keys())
        extra = sorted(actual_package.keys() - expected_package.keys())
        changed = sorted(
            relative
            for relative in actual_package.keys() & expected_package.keys()
            if actual_package[relative] != expected_package[relative]
        )
        raise ValueError(
            "staged package does not match the package that materialized the run: "
            f"missing={missing[:5]} extra={extra[:5]} changed={changed[:5]}"
        )
    for path in package_files:
        add_file(path, posix_join(repo_prefix, "src", str(path.relative_to(source_repo / "src"))), "package")

    # stage_identity binds every direct dependency receipt. Ship those exact
    # receipts before the provider recomputes the identity, or a provider stage
    # records ``None`` for dependencies that exist only on the host and its
    # receipt can never validate after returning.
    if receipts_dir is None:
        receipts_dir = plan_receipts_dir(plan)
    for dependency in sorted(dependencies):
        receipt_path = receipts_dir / f"{dependency}.json"
        if not receipt_path.is_file():
            raise ValueError(
                f"stage {stage['stage_id']} dependency receipt is missing: {receipt_path}"
            )
        dependency_receipt = load_json(receipt_path)
        if dependency_receipt.get("stage_id") != dependency:
            raise ValueError(
                f"stage {stage['stage_id']} dependency receipt names another stage: {dependency}"
            )
        if dependency_receipt.get("run_fingerprint") != plan.get("run_fingerprint"):
            raise ValueError(
                f"stage {stage['stage_id']} dependency receipt belongs to another run: {dependency}"
            )
        if dependency_receipt.get("ok") is not True:
            raise ValueError(
                f"stage {stage['stage_id']} dependency receipt is incomplete: {dependency}"
            )
        add_file(
            receipt_path,
            _volume_relative_path(str(receipt_path), policy.volume_mount),
            "dependency-receipt",
            expected_sha256=sha256_file(receipt_path),
            expected_bytes=receipt_path.stat().st_size,
            origin=f"dependency receipt {dependency}",
        )

    # Ship the upstream artifacts this stage declares. Declaring an input used to be a
    # reference check and nothing more: the file itself only reached the container when
    # the upstream stage had also run on Modal and written to the Volume. Every free
    # stage runs on the host, whose run root is a different store that happens to share
    # a path string with the mount, so a stage whose adapter really opens its declared
    # input found nothing there.
    shipped = {item["destination"] for item in files}
    input_files, skipped, input_errors = stage_input_files(
        plan, resolved_inputs, policy=policy, receipts_dir=receipts_dir
    )
    if input_errors:
        raise ValueError(
            f"stage {stage['stage_id']} cannot resolve its declared inputs: "
            + "; ".join(input_errors)
        )
    provenance: list[dict[str, Any]] = []
    for record in input_files:
        if record["destination"] in shipped:
            continue
        shipped.add(record["destination"])
        add_file(
            Path(record["source"]),
            record["destination"],
            "stage-input",
            expected_sha256=record.get("sha256"),
            expected_bytes=record.get("bytes"),
            origin=record.get("origin"),
        )
        provenance.append(
            {
                "destination": record["destination"],
                "origin": record.get("origin"),
                "verified_sha256": record.get("sha256") is not None,
                "verified_bytes": record.get("bytes") is not None,
                "digest_source": record.get("digest_source"),
            }
        )

    # An input that was not returned to the host is only admissible when the
    # upstream stage already wrote it to this Volume, which the job register is
    # the record of.
    # Reporting it and dispatching anyway is how an earlier run paid for a GPU job that
    # opened a file nobody had put there.
    if skipped:
        raise ValueError(
            f"stage {stage['stage_id']} declares inputs that were not returned to the host: "
            + "; ".join(f"{item['input_ref']} ({item['reason']})" for item in skipped)
        )

    destinations = [item["destination"] for item in files]
    if len(destinations) != len(set(destinations)):
        raise ValueError("bootstrap file list has duplicate Volume destinations")
    return {
        "schema_version": 1,
        "stage_id": stage["stage_id"],
        "adapter_id": adapter["adapter_id"],
        "stage_contract": {
            "path": str(contract_path),
            "contract_stage": contract_stages[stage["stage_id"]],
        },
        "stage_inputs": list(stage.get("inputs", [])),
        "stage_input_files": [record["destination"] for record in input_files],
        "stage_inputs_not_shipped": skipped,
        "stage_input_provenance": provenance,
        "volume_mappings": volumes_for_adapter(adapter, policy),
        "package_destination": posix_join(repo_prefix, "src", "claude_binder"),
        "files": files,
    }


def bootstrap_script(*, volume_mount: str, python_executable: str) -> str:
    """Build the one-shot Volume bootstrap script with exact package sync.

    The Volume persists beyond a run.  Extracting a new archive over its old
    package tree is therefore not a synchronization operation: a module that
    was deleted or renamed in the checkout can remain importable indefinitely.
    The script prunes *only* the declared package destination, then proves
    that its regular-file set and hashes exactly match the staged manifest.
    """
    return textwrap.dedent(
        f'''
        #!/usr/bin/env bash
        set -euo pipefail
        mkdir -p {shlex.quote(volume_mount)}
        {shlex.quote(python_executable)} - {BOOTSTRAP_MANIFEST} {shlex.quote(volume_mount)} <<'PY'
        import os
        import sys
        import json
        from pathlib import Path, PurePosixPath

        manifest = json.loads(Path(sys.argv[1]).read_text())
        mount = Path(sys.argv[2]).resolve()

        def fail(message):
            raise SystemExit("unsafe bootstrap manifest: " + message)

        def safe_parts(value, label):
            if not isinstance(value, str) or not value:
                fail(label + " must be a non-empty relative path")
            if value.startswith("/") or any(part in {{"", ".", ".."}} for part in value.split("/")):
                fail(label + " is not a clean relative path: " + value)
            parts = PurePosixPath(value).parts
            if not parts:
                fail(label + " has no path components")
            return parts

        package_parts = safe_parts(manifest.get("package_destination"), "package_destination")
        package_relative = PurePosixPath(*package_parts)
        files = manifest.get("files")
        if not isinstance(files, list):
            fail("files must be a list")

        # Do not traverse an existing symlink on the route to the sole
        # directory this bootstrap is allowed to prune.  A mounted Volume may
        # be persistent or shared, so following one would make a stale-package
        # cleanup capable of touching an unrelated tree.
        package_root = mount
        for index, part in enumerate(package_parts):
            package_root = package_root / part
            if package_root.is_symlink():
                fail("package destination contains a symlink: " + str(package_root))
            if index < len(package_parts) - 1 and package_root.exists() and not package_root.is_dir():
                fail("package destination parent is not a directory: " + str(package_root))

        expected_files = set()
        expected_dirs = set()
        for item in files:
            if not isinstance(item, dict):
                fail("file entry is not an object")
            destination = item.get("destination")
            destination_parts = safe_parts(destination, "file destination")
            destination_path = PurePosixPath(*destination_parts)
            if item.get("kind") != "package":
                continue
            try:
                relative = destination_path.relative_to(package_relative)
            except ValueError:
                fail("package file is outside package_destination: " + destination)
            if not relative.parts:
                fail("package file names the package directory itself")
            relative_name = relative.as_posix()
            if relative_name in expected_files:
                fail("duplicate package destination: " + destination)
            expected_files.add(relative_name)
            for parent in relative.parents:
                if parent.parts:
                    expected_dirs.add(parent.as_posix())

        if not expected_files:
            fail("manifest declares no package files")

        if package_root.exists() and not package_root.is_dir():
            fail("package destination is not a directory: " + str(package_root))
        if package_root.exists():
            for current, directories, filenames in os.walk(package_root, topdown=False, followlinks=False):
                current_path = Path(current)
                relative_current = current_path.relative_to(package_root)
                for name in filenames:
                    path = current_path / name
                    relative_name = (relative_current / name).as_posix()
                    # Removing a link does not follow it.  It must go even if
                    # its name is expected, otherwise tar could write through
                    # it into an arbitrary mounted path.
                    if path.is_symlink() or relative_name not in expected_files:
                        path.unlink()
                for name in directories:
                    path = current_path / name
                    relative_name = (relative_current / name).as_posix()
                    if path.is_symlink():
                        path.unlink()
                    elif relative_name not in expected_dirs:
                        path.rmdir()
        package_root.mkdir(parents=True, exist_ok=True)
        PY
        tar -xzf {BOOTSTRAP_ARCHIVE} -C {shlex.quote(volume_mount)}
        {shlex.quote(python_executable)} - {BOOTSTRAP_MANIFEST} {shlex.quote(volume_mount)} <<'PY'
        import hashlib
        import json
        import os
        import sys
        from pathlib import Path, PurePosixPath

        manifest = json.loads(Path(sys.argv[1]).read_text())
        mount = Path(sys.argv[2]).resolve()

        def safe_parts(value, label):
            if not isinstance(value, str) or not value:
                raise SystemExit(label + " must be a non-empty relative path")
            if value.startswith("/") or any(part in {{"", ".", ".."}} for part in value.split("/")):
                raise SystemExit(label + " is not a clean relative path: " + value)
            parts = PurePosixPath(value).parts
            if not parts:
                raise SystemExit(label + " has no path components")
            return parts

        package_relative = PurePosixPath(*safe_parts(manifest.get("package_destination"), "package_destination"))
        package_root = mount.joinpath(*package_relative.parts)
        if package_root.is_symlink() or not package_root.is_dir():
            raise SystemExit("package destination is not a regular directory: " + str(package_root))

        errors = []
        expected_package_files = set()
        for item in manifest.get("files", []):
            if not isinstance(item, dict):
                errors.append("bootstrap manifest has a non-object file entry")
                continue
            destination = item.get("destination")
            try:
                destination_path = PurePosixPath(*safe_parts(destination, "file destination"))
            except SystemExit as exc:
                errors.append(str(exc))
                continue
            path = mount.joinpath(*destination_path.parts)
            if not path.is_file():
                errors.append("missing bootstrapped file: " + str(path))
                continue
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            if digest != item["sha256"]:
                errors.append("bootstrap hash mismatch: " + str(path))
            if item.get("kind") == "package":
                try:
                    relative = destination_path.relative_to(package_relative)
                except ValueError:
                    errors.append("package file is outside package_destination: " + str(destination))
                else:
                    if relative.parts:
                        expected_package_files.add(relative.as_posix())

        actual_package_files = set()
        for current, directories, filenames in os.walk(package_root, topdown=True, followlinks=False):
            current_path = Path(current)
            relative_current = current_path.relative_to(package_root)
            for name in directories:
                if (current_path / name).is_symlink():
                    errors.append("package directory is a symlink: " + str(current_path / name))
            for name in filenames:
                path = current_path / name
                if path.is_symlink():
                    errors.append("package file is a symlink: " + str(path))
                else:
                    actual_package_files.add((relative_current / name).as_posix())
        missing_package = sorted(expected_package_files - actual_package_files)
        extra_package = sorted(actual_package_files - expected_package_files)
        if missing_package:
            errors.append("missing package files after bootstrap: " + ", ".join(missing_package))
        if extra_package:
            errors.append("unexpected package files after bootstrap: " + ", ".join(extra_package))
        if errors:
            raise SystemExit("\\n".join(errors))
        result = {{
            "ok": True,
            "file_count": len(manifest["files"]),
            "package_file_count": len(expected_package_files),
            "package_destination": package_relative.as_posix(),
        }}
        Path("out/bootstrap-result.json").write_text(json.dumps(result) + "\\n")
        print(json.dumps(result))
        PY
        '''
    )


def build_bootstrap_spec(
    plan: dict[str, Any],
    stage: dict[str, Any],
    adapter: dict[str, Any],
    *,
    bundle_root: Path,
    source_repo: Path,
    policy: DispatchPolicy,
    attempt_id: str,
    receipts_dir: Path | None = None,
    register: Sequence[dict[str, Any]] | None = None,
) -> JobSpec:
    """Build the reviewable job that populates the run Volume.

    ``attempt_id`` is deliberately required.  The bootstrap is the opening
    phase of the paid stage attempt, not a second pseudo-attempt: sharing the
    identifier makes ``submit_wave`` reuse the stage's one approved reservation
    for bootstrap, smoke/scale, and finalization.
    """
    if not policy.python_executable or UNRESOLVED in policy.python_executable:
        raise ValueError("dispatch policy must resolve python_executable before bootstrap")
    manifest = bootstrap_file_manifest(
        plan,
        stage,
        adapter,
        bundle_root=bundle_root,
        source_repo=source_repo,
        policy=policy,
        receipts_dir=receipts_dir,
        register=register,
    )
    params = provider_params(
        adapter,
        volumes=manifest["volume_mappings"],
        container_timeout_s=policy.container_timeout_s,
        cpu_only=True,
    )
    return JobSpec(
        stage_id=stage["stage_id"],
        adapter_id=adapter["adapter_id"],
        phase="bootstrap",
        attempt_id=attempt_id,
        shard=None,
        intent=f"{plan.get('campaign_id', 'campaign')}: bootstrap run bundle and repository subset",
        command=f"bash {BOOTSTRAP_SCRIPT}",
        script_text=bootstrap_script(
            volume_mount=policy.volume_mount,
            python_executable=policy.python_executable,
        ),
        inputs=[],
        outputs=["bootstrap-result.json"],
        run_timeout_s=paid_job_timeout_s(
            min(1800, int(stage["timeout_minutes"]) * 60),
            label=f"bootstrap job timeout for stage {stage['stage_id']}",
        ),
        provider_params=params,
        receipt_script_text="",
        bootstrap_files=manifest["files"],
        bootstrap_package_destination=manifest["package_destination"],
    )


def repo_bootstrap_note(policy: DispatchPolicy) -> str:
    """Describe the contract-driven bootstrap required before a stage job."""
    volume = policy.volume_name or "<unset, see dispatch-policy.json>"
    return (
        f"Plan the first Modal stage, then build and submit build_bootstrap_spec "
        f"once with that plan's attempt_id. It verifies the stage "
        f"contract, copies the run bundle and package subset to {volume}:{policy.repo_dir}, "
        "and hashes every file after extraction. The stage input references "
        "remain on that same Volume so every dependency receipt and artifact "
        "keeps its recorded path."
    )


class WaveSubmitFailed(RuntimeError):
    """A wave stopped partway, carrying the jobs that did reach the provider.

    A handle exists only as a Python object in the submitting cell. Letting an
    exception discard the list of submitted rows discards the only close path
    those handles have, and each one holds a container that bills idle until it
    is closed (SKILL.md:466-472). Worse, an unclosed handle is also a missing
    Volume commit, which is the barrier the next wave depends on.

    So the partial wave rides out on the exception. Catch this, collect the
    notifications for ``exc.submitted``, and run ``close_wave`` on it before
    deciding anything about the specs that never got submitted.
    """

    def __init__(self, message: str, submitted: list[dict[str, Any]]) -> None:
        super().__init__(message)
        self.submitted = submitted


def _dispatch_approval_check(
    plan: dict[str, Any], *, bundle_root: Path | None = None
) -> dict[str, Any]:
    """Return a refusal for any malformed dispatch approval input."""
    stage_contract_path = plan.get("stage_contract_path")
    if not isinstance(stage_contract_path, str) or not stage_contract_path:
        return {
            "ok": False,
            "errors": ["the plan has no valid stage contract path"],
        }
    runtime = plan.get("runtime")
    if not isinstance(runtime, dict):
        return {
            "ok": False,
            "errors": ["the plan has no valid runtime bundle path"],
        }
    bundle_path = runtime.get("bundle_path")
    if not isinstance(bundle_path, str) or not bundle_path:
        return {
            "ok": False,
            "errors": ["the plan has no valid runtime bundle path"],
        }
    runtime_bundle_root = Path(bundle_path)
    if not runtime_bundle_root.is_absolute():
        return {
            "ok": False,
            "errors": ["the plan runtime bundle path is not absolute"],
        }
    contract_path = Path(stage_contract_path)
    if not contract_path.is_absolute():
        return {
            "ok": False,
            "errors": ["the plan stage contract path is not absolute"],
        }
    if contract_path.resolve() != (runtime_bundle_root / "stage-contract.json").resolve():
        return {
            "ok": False,
            "errors": [
                "the plan stage contract path does not match its runtime bundle path"
            ],
        }
    approval_root = bundle_root or runtime_bundle_root
    if not approval_root.is_absolute():
        return {
            "ok": False,
            "errors": ["the local approval bundle path is not absolute"],
        }
    try:
        return verify_execution_approval(plan, approval_root)
    except Exception as exc:
        return {
            "ok": False,
            "errors": [f"the approval check failed closed: {type(exc).__name__}: {exc}"],
        }


def validate_job_spec_for_paid_submit(spec: JobSpec) -> list[str]:
    """Return execution hazards that must block a paid submit."""
    errors = list(spec.unresolved_paths)
    try:
        paid_job_timeout_s(
            spec.run_timeout_s,
            label=f"stage {spec.stage_id} {spec.phase} job timeout",
        )
    except ValueError as exc:
        errors.append(str(exc))
    serialized = json.dumps(spec.as_dict(), sort_keys=True)
    if UNRESOLVED in serialized:
        errors.append("job specification still contains the __REQUIRED__ placeholder")
    if any(
        UNRESOLVED in value
        for value in (spec.command, spec.script_text, spec.receipt_script_text)
    ):
        errors.append("job command or staged script still contains the __REQUIRED__ placeholder")
    for key, value in spec.provider_params.items():
        if isinstance(value, str) and (UNRESOLVED in value or not value):
            errors.append(f"provider parameter {key} is unresolved")
    return sorted(set(errors))


def _provider_for_budget_gate(plan: dict[str, Any], bundle_root: Path | None) -> dict[str, Any]:
    """Read the materialized provider budget, including plans made before it entered the plan."""
    provider = plan.get("provider")
    if isinstance(provider, dict):
        return provider
    if bundle_root is None:
        return {}
    config_path = bundle_root / "config.resolved.json"
    if not config_path.is_file():
        return {}
    config = load_json(config_path)
    provider = config.get("provider")
    return provider if isinstance(provider, dict) else {}


def _approved_stage_estimates(approval: dict[str, Any]) -> dict[str, Any]:
    """Return the pre-run estimator's maximum USD estimate for each approved stage."""
    from claude_binder.lane import approved_stage_estimates

    return approved_stage_estimates(approval)


def _dispatch_authorization_check(
    plan: dict[str, Any],
    *,
    stage_id: str,
    run_root: Path,
    bundle_root: Path | None,
    modal_details_reader: Any,
    modal_identity_reader: Any = None,
) -> dict[str, Any]:
    """Re-measure Modal authorization at the handle-creation boundary."""
    if modal_details_reader is None:
        return {
            "ok": False,
            "errors": [
                "Modal authorization was not measured: pass the live compute_details "
                "reader from the Claude Science kernel"
            ],
        }
    stage = next(
        (
            item
            for item in plan.get("stages", [])
            if isinstance(item, dict) and item.get("stage_id") == stage_id
        ),
        None,
    )
    if stage is None:
        return {"ok": False, "errors": [f"plan has no stage {stage_id!r}"]}
    if bundle_root is None:
        return {
            "ok": False,
            "errors": ["the local run-bundle path is required to measure Modal authorization"],
        }
    from claude_binder.lane import provider_authorization_preflight

    return provider_authorization_preflight(
        plan,
        [stage],
        base_context={
            "config_path": str(bundle_root / "config.resolved.json"),
            "plan_path": str(bundle_root / "run-plan.json"),
        },
        run_root=run_root,
        progress_path=run_root / "progress.jsonl",
        modal_details_reader=modal_details_reader,
        modal_identity_reader=modal_identity_reader,
    )


def _wave_identity(specs: Sequence[JobSpec]) -> tuple[str, str]:
    if not specs:
        raise ValueError("a paid wave must contain at least one job specification")
    identities = {(spec.stage_id, spec.attempt_id) for spec in specs}
    if len(identities) != 1:
        raise ValueError("one paid wave must contain exactly one stage and attempt")
    stage_id, attempt_id = next(iter(identities))
    if not attempt_id:
        raise ValueError("a paid wave must name a non-empty attempt_id")
    phases = {spec.phase for spec in specs}
    if len(phases) != 1:
        raise ValueError("one paid wave must contain exactly one phase")
    keys = [_spec_dispatch_key(spec) for spec in specs]
    if len(keys) != len(set(keys)):
        raise ValueError("a paid wave repeats the same phase/shard specification")
    return stage_id, attempt_id


def _require_phase_barrier(
    run_root: Path,
    *,
    run_fingerprint: str,
    stage_id: str,
    attempt_id: str,
    phase: str,
    workspace: Path | None = None,
) -> None:
    """Refuse a phase that races or bypasses the prior wave's close barrier.

    ``workspace`` is where the harvest lands, and the scale phase needs it. A
    smoke job that exits 0 having produced nothing satisfies every other check
    here, so scale is refused unless the harvested phase validation proves the
    smoke phase counted outputs and parsed one. Omitting ``workspace`` refuses
    scale rather than skipping the check.
    """
    register_path = run_root / "artifacts" / JOB_REGISTER
    rows = load_jsonl(register_path) if register_path.is_file() else []
    intents = [
        row
        for row in rows
        if row.get("record") in {"submission-intent", "submission-uncertain"}
        and row.get("run_fingerprint") == run_fingerprint
        and row.get("stage_id") == stage_id
        and row.get("attempt_id") == attempt_id
    ]
    bound_submission_ids = {
        row.get("submission_id")
        for row in rows
        if row.get("job_id") and row.get("submission_id")
    }
    ambiguous = [
        row.get("submission_id")
        for row in intents
        if row.get("submission_id") not in bound_submission_ids
    ]
    if ambiguous:
        raise ValueError(
            "a prior submit has no bound job id; inspect the provider ledger before "
            f"continuing: {', '.join(str(item) for item in ambiguous)}"
        )
    jobs = [
        row
        for row in merge_register(rows)
        if row.get("run_fingerprint") == run_fingerprint
        and row.get("stage_id") == stage_id
        and row.get("attempt_id") == attempt_id
    ]
    unfinished = [
        row.get("job_id")
        for row in jobs
        if row.get("state") == "submitted" or row.get("close_state") != "closed"
    ]
    if unfinished:
        raise ValueError(
            "the previous wave has not crossed its close barrier: "
            + ", ".join(str(item) for item in unfinished)
        )
    failed = [
        row.get("job_id")
        for row in jobs
        if row.get("state") != "succeeded" or row.get("exit_code") != 0
    ]
    if failed:
        raise ValueError(
            "the prior wave failed; start a separately authorized retry attempt: "
            + ", ".join(str(item) for item in failed)
        )
    prior_phases = {str(row.get("phase")) for row in jobs}
    if phase == "bootstrap" and jobs:
        raise ValueError("bootstrap must be the first and only bootstrap wave")
    if phase in {"single", "smoke"} and prior_phases - {"bootstrap"}:
        raise ValueError(
            f"phase {phase} must be the first scientific wave after an optional bootstrap"
        )
    if phase == "scale":
        if "smoke" not in prior_phases:
            raise ValueError("scale refused until the smoke job succeeds and closes")
        if workspace is None:
            raise ValueError(
                "scale refused: checking the smoke evidence needs the workspace the "
                "harvest wrote to, and this call supplied none"
            )
        for row in jobs:
            if row.get("phase") != "smoke":
                continue
            evidence = smoke_evidence(
                workspace,
                [row],
                str(row.get("job_id")),
                run_fingerprint=run_fingerprint,
                stage_id=stage_id,
            )
            if not evidence["ok"]:
                raise ValueError(
                    "scale refused: the smoke job returned no host-side evidence that "
                    "it scored anything: " + "; ".join(evidence["errors"])
                )
    if phase == "finalize" and "scale" not in prior_phases:
        raise ValueError("finalize refused until every scale shard succeeds and closes")
    if phase not in {"bootstrap", "single", "smoke", "scale", "finalize"}:
        raise ValueError(f"unsupported paid wave phase: {phase}")


def _spec_dispatch_key(spec: JobSpec) -> tuple[Any, ...]:
    shard = spec.shard.as_dict() if spec.shard else None
    return (spec.stage_id, spec.attempt_id, spec.phase, json.dumps(shard, sort_keys=True))


def _reservation_id(run_fingerprint: str, stage_id: str, attempt_id: str) -> str:
    return sha256_json(
        {
            "kind": "modal-dispatch-reservation",
            "run_fingerprint": run_fingerprint,
            "stage_id": stage_id,
            "attempt_id": attempt_id,
        }
    )


def _reservation_rows(
    spend_rows: Sequence[dict[str, Any]],
    *,
    run_fingerprint: str,
    stage_id: str,
    attempt_id: str,
) -> list[dict[str, Any]]:
    return [
        row
        for row in spend_rows
        if row.get("run_fingerprint") == run_fingerprint
        and row.get("stage_id") == stage_id
        and row.get("event") == "charge-estimate"
        and isinstance(row.get("details"), dict)
        and row["details"].get("dispatch_reservation") is True
        and row["details"].get("attempt_id") == attempt_id
    ]


def _validated_retry_authorizations(run_root: Path) -> list[dict[str, Any]]:
    path = run_root / "artifacts" / RETRY_AUTHORIZATION_REGISTER
    rows = load_jsonl(path) if path.is_file() else []
    previous_hash: str | None = None
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"retry authorization row {index} is not an object")
        declared_previous = row.get("previous_row_sha256")
        if declared_previous != previous_hash:
            raise ValueError(f"retry authorization row {index} breaks the hash chain")
        payload = {key: value for key, value in row.items() if key != "row_sha256"}
        if row.get("row_sha256") != sha256_json(payload):
            raise ValueError(f"retry authorization row {index} has an invalid hash")
        previous_hash = row["row_sha256"]
    return rows


def _reservation_is_reconciled(
    spend_rows: Sequence[dict[str, Any]], reservation_id: str
) -> bool:
    return any(
        row.get("event") == "charge"
        and isinstance(row.get("details"), dict)
        and row["details"].get("dispatch_reservation_reconciliation") is True
        and row["details"].get("reservation_id") == reservation_id
        for row in spend_rows
    )


def _require_retry_authorization(
    run_root: Path,
    *,
    run_fingerprint: str,
    stage_id: str,
    attempt_id: str,
) -> None:
    register_path = run_root / "artifacts" / JOB_REGISTER
    jobs = load_jsonl(register_path) if register_path.is_file() else []
    prior_attempts = {
        str(row.get("attempt_id"))
        for row in jobs
        if row.get("run_fingerprint") == run_fingerprint
        and row.get("stage_id") == stage_id
        and isinstance(row.get("attempt_id"), str)
        and row.get("attempt_id") != attempt_id
    }
    if not prior_attempts:
        return
    authorizations = _validated_retry_authorizations(run_root)
    matching = [
        row
        for row in authorizations
        if row.get("run_fingerprint") == run_fingerprint
        and row.get("stage_id") == stage_id
        and row.get("new_attempt_id") == attempt_id
        and row.get("prior_attempt_id") in prior_attempts
    ]
    if not matching:
        raise ValueError(
            f"retry attempt {stage_id}/{attempt_id} has no intact append-only authorization"
        )
    spend_path = run_root / "artifacts" / "spend.jsonl"
    spend_rows = load_jsonl(spend_path) if spend_path.is_file() else []
    prior_attempt_id = str(matching[-1]["prior_attempt_id"])
    prior_reservations = _reservation_rows(
        spend_rows,
        run_fingerprint=run_fingerprint,
        stage_id=stage_id,
        attempt_id=prior_attempt_id,
    )
    if not prior_reservations or not _reservation_is_reconciled(
        spend_rows, str(prior_reservations[-1]["details"]["reservation_id"])
    ):
        raise ValueError(
            f"retry attempt {stage_id}/{attempt_id} is authorized, but prior attempt "
            f"{prior_attempt_id} has not reached its financial barrier"
        )


def authorize_retry(
    run_root: Path,
    *,
    plan: dict[str, Any],
    stage_id: str,
    prior_attempt_id: str,
    new_attempt_id: str,
    authorized_by: str,
    reason: str,
) -> dict[str, Any]:
    """Append one hash-chained authorization for a new paid attempt.

    This only records an operator decision; it never creates a provider handle.
    The prior attempt must already be closed and financially reconciled.
    """
    for label, value in (
        ("stage_id", stage_id),
        ("prior_attempt_id", prior_attempt_id),
        ("new_attempt_id", new_attempt_id),
        ("authorized_by", authorized_by),
        ("reason", reason),
    ):
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{label} must be a non-empty string")
    if prior_attempt_id == new_attempt_id:
        raise ValueError("a retry must use a new attempt_id")
    run_fingerprint = plan.get("run_fingerprint")
    if not isinstance(run_fingerprint, str) or not run_fingerprint:
        raise ValueError("the plan has no run_fingerprint")
    with dispatch_lock(run_root):
        register_path = run_root / "artifacts" / JOB_REGISTER
        merged = merge_register(
            load_jsonl(register_path) if register_path.is_file() else []
        )
        prior_jobs = [
            row
            for row in merged
            if row.get("run_fingerprint") == run_fingerprint
            and row.get("stage_id") == stage_id
            and row.get("attempt_id") == prior_attempt_id
        ]
        if not prior_jobs:
            raise ValueError(
                f"prior paid attempt {stage_id}/{prior_attempt_id} is not recorded"
            )
        unsafe = [
            row.get("job_id")
            for row in prior_jobs
            if row.get("state") == "submitted" or row.get("close_state") != "closed"
        ]
        if unsafe:
            raise ValueError(
                "retry authorization refused until every prior handle is terminal and "
                f"closed: {', '.join(str(item) for item in unsafe)}"
            )
        spend_path = run_root / "artifacts" / "spend.jsonl"
        spend_rows = load_jsonl(spend_path) if spend_path.is_file() else []
        reservations = _reservation_rows(
            spend_rows,
            run_fingerprint=run_fingerprint,
            stage_id=stage_id,
            attempt_id=prior_attempt_id,
        )
        if not reservations:
            raise ValueError("prior paid attempt has no dispatch reservation")
        reservation_id = str(reservations[-1]["details"]["reservation_id"])
        if not _reservation_is_reconciled(spend_rows, reservation_id):
            raise ValueError("prior paid attempt has not reached its financial barrier")

        rows = _validated_retry_authorizations(run_root)
        if any(
            row.get("run_fingerprint") == run_fingerprint
            and row.get("stage_id") == stage_id
            and row.get("new_attempt_id") == new_attempt_id
            for row in rows
        ):
            raise ValueError(f"retry attempt_id is already authorized: {new_attempt_id}")
        row = {
            "schema_version": 1,
            "timestamp": utc_now(),
            "run_fingerprint": run_fingerprint,
            "run_id": plan.get("run_id"),
            "stage_id": stage_id,
            "prior_attempt_id": prior_attempt_id,
            "new_attempt_id": new_attempt_id,
            "authorized_by": authorized_by.strip(),
            "reason": reason.strip(),
            "previous_row_sha256": rows[-1]["row_sha256"] if rows else None,
        }
        row["row_sha256"] = sha256_json(row)
        append_jsonl(run_root / "artifacts" / RETRY_AUTHORIZATION_REGISTER, row)
        return row


def bind_uncertain_submission(
    run_root: Path,
    *,
    plan: dict[str, Any],
    submission_id: str,
    job_id: str,
) -> dict[str, Any]:
    """Bind a provider-ledger job ID to one durable uncertain submit intent."""
    if not isinstance(submission_id, str) or not submission_id.strip():
        raise ValueError("submission_id must be a non-empty string")
    if not isinstance(job_id, str) or not job_id.strip():
        raise ValueError("job_id must be a non-empty string")
    run_fingerprint = plan.get("run_fingerprint")
    if not isinstance(run_fingerprint, str) or not run_fingerprint:
        raise ValueError("the plan has no run_fingerprint")
    register_path = run_root / "artifacts" / JOB_REGISTER
    with dispatch_lock(run_root):
        rows = load_jsonl(register_path) if register_path.is_file() else []
        if any(row.get("job_id") == job_id for row in rows):
            raise ValueError(f"job_id is already registered: {job_id}")
        if any(
            row.get("submission_id") == submission_id and row.get("job_id")
            for row in rows
        ):
            raise ValueError(f"submission_id is already bound: {submission_id}")
        matches = [
            row
            for row in rows
            if row.get("record") in {"submission-intent", "submission-uncertain"}
            and row.get("submission_id") == submission_id
            and row.get("run_fingerprint") == run_fingerprint
        ]
        if not matches:
            raise ValueError("no uncertain submission intent matches this run and ID")
        intent = matches[-1]
        row = {
            key: intent.get(key)
            for key in (
                "run_fingerprint",
                "run_id",
                "stage_id",
                "adapter_id",
                "phase",
                "attempt_id",
                "shard",
                "intent",
                "run_timeout_s",
                "provider_params",
                "reservation_id",
                "submission_id",
            )
        }
        row.update(
            {
                "job_id": job_id.strip(),
                "submitted_at": utc_now(),
                "state": "submitted",
                "recovered_from_provider_ledger": True,
            }
        )
        append_jsonl(register_path, row)
        return row


def submit_wave(
    host: Any,
    specs: Sequence[JobSpec],
    *,
    workspace: Path,
    run_root: Path,
    policy: DispatchPolicy,
    plan: dict[str, Any],
    bundle_root: Path | None = None,
    modal_details_reader: Any = None,
    modal_identity_reader: Any = None,
) -> list[dict[str, Any]]:
    """Submit every job in one wave, then return without waiting.

    One handle is one container, and submits on a handle are strictly
    sequential because each one wipes the working directory. Parallelism is
    therefore N handles, one per shard, all created and submitted in this one
    call. That is what the job surface is built for.

    Nothing here waits. ``job.result()`` never blocks, and calling it before the
    notification arrives raises ``JobPending``. The caller ends the cell after
    this returns and parks on the ``wait_for_notification`` brain tool.

    Every submitted job is written to the job register before this returns. A
    session that dies mid-wave can then tell a shard that never ran from one
    whose job is still going, which is the difference between resuming and
    paying twice.

    Reading the previous wave's Volume writes is safe from here. One handle
    carries one job, so every job gets its own fresh container, and a fresh
    container mounts the latest state of an attached Volume at creation
    (volume_visibility.md:65-68). No reload is needed. This holds only while one
    handle carries one job. Two jobs on one handle would leave the second with
    the first's mount.

    Writing is the half that needs the close, and nothing in this function
    completes a wave. See ``close_wave`` for why, and call it.

    A failure partway through raises ``WaveSubmitFailed``, which carries the
    rows already submitted. Close them.
    """
    try:
        stage_id, attempt_id = _wave_identity(specs)
    except ValueError as exc:
        raise WaveSubmitFailed(f"paid Modal submit refused: {exc}", []) from exc
    approval_check = _dispatch_approval_check(plan, bundle_root=bundle_root)
    if approval_check.get("ok") is not True:
        errors = approval_check.get("errors")
        if not isinstance(errors, list) or not errors:
            errors = ["the approval gate refused dispatch"]
        raise WaveSubmitFailed(
            "approval gate refused: " + "; ".join(str(error) for error in errors),
            [],
        )

    unresolved = [
        f"{spec.stage_id}/{spec.phase}: {error}"
        for spec in specs
        for error in validate_job_spec_for_paid_submit(spec)
    ]
    if unresolved:
        raise WaveSubmitFailed(
            "paid Modal submit refused before handle creation: " + "; ".join(unresolved),
            [],
        )
    try:
        staged_inputs = {
            _spec_dispatch_key(spec): stage_inputs(
                spec, workspace=workspace, policy=policy
            )
            for spec in specs
        }
    except Exception as exc:
        raise WaveSubmitFailed(
            f"paid Modal submit refused during free local staging: {type(exc).__name__}: {exc}",
            [],
        ) from exc

    register_path = run_root / "artifacts" / JOB_REGISTER
    spend_path = run_root / "artifacts" / "spend.jsonl"
    submitted: list[dict[str, Any]] = []
    try:
        with dispatch_lock(run_root):
            authorization = _dispatch_authorization_check(
                plan,
                stage_id=stage_id,
                run_root=run_root,
                bundle_root=bundle_root,
                modal_details_reader=modal_details_reader,
                modal_identity_reader=modal_identity_reader,
            )
            if authorization.get("ok") is not True:
                errors = authorization.get("errors")
                if not isinstance(errors, list) or not errors:
                    errors = ["Modal authorization was not measured as authorized"]
                raise ValueError("; ".join(str(error) for error in errors))
            provider = _provider_for_budget_gate(plan, bundle_root)
            budget_value = provider.get("budget")
            budget = budget_value if isinstance(budget_value, dict) else {}
            open_run_registers(
                run_root,
                {"provider": provider},
                run_fingerprint=plan["run_fingerprint"],
            )
            existing_jobs = load_jsonl(register_path) if register_path.is_file() else []
            existing_keys = {
                (
                    row.get("stage_id"),
                    row.get("attempt_id"),
                    row.get("phase"),
                    json.dumps(row.get("shard"), sort_keys=True),
                )
                for row in existing_jobs
                if row.get("record") in {None, "submission-intent", "submission-uncertain"}
            }
            duplicates = [spec for spec in specs if _spec_dispatch_key(spec) in existing_keys]
            if duplicates:
                duplicate = duplicates[0]
                raise ValueError(
                    f"job {duplicate.stage_id}/{duplicate.attempt_id}/{duplicate.phase} "
                    "is already registered; attach or resume it instead of resubmitting"
                )
            _require_retry_authorization(
                run_root,
                run_fingerprint=plan["run_fingerprint"],
                stage_id=stage_id,
                attempt_id=attempt_id,
            )
            _require_phase_barrier(
                run_root,
                run_fingerprint=plan["run_fingerprint"],
                stage_id=stage_id,
                attempt_id=attempt_id,
                phase=specs[0].phase,
                workspace=workspace,
            )

            approved = _approved_stage_estimates(approval_check.get("approval", {}))
            estimate = approved.get(stage_id)
            reservation_key = f"{stage_id}@{attempt_id}"
            spend_rows = load_jsonl(spend_path) if spend_path.is_file() else []
            reservations = _reservation_rows(
                spend_rows,
                run_fingerprint=plan["run_fingerprint"],
                stage_id=stage_id,
                attempt_id=attempt_id,
            )
            reservation_id = _reservation_id(
                plan["run_fingerprint"], stage_id, attempt_id
            )
            if reservations:
                reservation = reservations[-1]
                if (
                    reservation.get("currency") != budget.get("currency")
                    or float(reservation.get("amount", -1)) != float(estimate)
                    or reservation.get("details", {}).get("reservation_id")
                    != reservation_id
                ):
                    raise ValueError("the existing dispatch reservation disagrees with approval")
                if _reservation_is_reconciled(spend_rows, reservation_id):
                    raise ValueError(
                        f"paid attempt {stage_id}/{attempt_id} is already financially closed"
                    )
            else:
                budget_check = enforce_spend_cap(
                    spend_path,
                    run_fingerprint=plan["run_fingerprint"],
                    budget_maximum=budget.get("maximum_spend_usd"),
                    currency=budget.get("currency"),
                    stage_estimates={reservation_key: estimate},
                    stage_ids=[reservation_key],
                )
                if budget_check.get("ok") is not True:
                    errors = budget_check.get("errors")
                    if not isinstance(errors, list) or not errors:
                        errors = ["the spend cap could not authorize this paid dispatch"]
                    raise ValueError("; ".join(str(error) for error in errors))
                reservation = record_spend(
                    spend_path,
                    run_fingerprint=plan["run_fingerprint"],
                    amount=float(estimate),
                    currency=budget.get("currency"),
                    amount_source="approved pre-run stage maximum",
                    event="charge-estimate",
                    stage_id=stage_id,
                    provider_id="modal",
                    budget_maximum=budget.get("maximum_spend_usd"),
                    details={
                        "dispatch_reservation": True,
                        "reservation_id": reservation_id,
                        "attempt_id": attempt_id,
                        "approved_stage_maximum": float(estimate),
                    },
                    note="reserved the full approved stage maximum before handle creation",
                )

            for spec in specs:
                params = spec.provider_params
                handle = host.compute.create("modal", provider_params=params)
                slots = getattr(handle, "concurrency", None)
                if slots is not None and slots.limit is not None and slots.live >= slots.limit:
                    raise RuntimeError(
                        f"the session concurrency cap is full at {slots.live} of "
                        f"{slots.limit}; collect a running job before submitting "
                        f"{spec.intent!r}"
                    )
                submission_id = str(uuid.uuid4())
                intent_row = {
                    "record": "submission-intent",
                    "submission_id": submission_id,
                    "run_fingerprint": plan["run_fingerprint"],
                    "run_id": plan["run_id"],
                    "stage_id": spec.stage_id,
                    "adapter_id": spec.adapter_id,
                    "phase": spec.phase,
                    "attempt_id": spec.attempt_id,
                    "shard": spec.shard.as_dict() if spec.shard else None,
                    "intent": spec.intent,
                    "run_timeout_s": spec.run_timeout_s,
                    "provider_params": params,
                    "reservation_id": reservation_id,
                    "intent_recorded_at": utc_now(),
                }
                append_jsonl(register_path, intent_row)
                try:
                    job = handle.submit_job(
                        intent=spec.intent,
                        command=spec.command,
                        inputs=staged_inputs[_spec_dispatch_key(spec)],
                        outputs=spec.outputs,
                        run_timeout_s=spec.run_timeout_s,
                    )
                except Exception as exc:
                    append_jsonl(
                        register_path,
                        {
                            **intent_row,
                            "record": "submission-uncertain",
                            "submit_error": f"{type(exc).__name__}: {exc}",
                            "uncertain_at": utc_now(),
                        },
                    )
                    raise
                row = {
                    "run_fingerprint": plan["run_fingerprint"],
                    "run_id": plan["run_id"],
                    "job_id": job.id,
                    "stage_id": spec.stage_id,
                    "adapter_id": spec.adapter_id,
                    "phase": spec.phase,
                    "attempt_id": spec.attempt_id,
                    "shard": spec.shard.as_dict() if spec.shard else None,
                    "intent": spec.intent,
                    "run_timeout_s": spec.run_timeout_s,
                    "provider_params": params,
                    "reservation_id": reservation_id,
                    "submission_id": submission_id,
                    "submitted_at": utc_now(),
                    "state": "submitted",
                }
                append_jsonl(register_path, row)
                submitted.append(dict(row, handle=handle, job=job))
    except Exception as exc:
        prefix = (
            "budget gate refused before handle creation"
            if not submitted and "budget cap" in str(exc)
            else f"the wave stopped after {len(submitted)} of {len(specs)} submits"
        )
        raise WaveSubmitFailed(
            f"{prefix}, and submitted handles ride out on this exception: "
            f"{type(exc).__name__}: {exc}",
            submitted,
        ) from exc
    return submitted


def record_fanout(run_root: Path, resolved: dict[str, Any]) -> None:
    """Publish a resolved fan-out width the moment it exists.

    Two readers need this. A resumed session needs it to know how many shards a
    stage had. A person watching the run needs it because it is the only
    denominator they ever get: the width was unknowable when the plan was
    approved, so until a stage resolves there is no number to compare progress
    against.
    """
    append_jsonl(run_root / "artifacts" / FANOUT_REGISTER, resolved)


def collect_notifications(
    run_root: Path, notifications: Iterable[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Fold one batch of compute_done payloads into the job register.

    One ``wait_for_notification`` call can return several entries at once, so
    iterate the list rather than assuming one. The loop ends when the tool
    returns ``{status: 'error'}``, which means no compute jobs remain.

    The payload carries ``job_id``, ``state``, ``exit_code``, ``notes`` and
    ``output_files``, plus ``error_kind``, ``system_hint`` and ``deadline_fired``
    when they are set. That is enough to decide what to do next without
    re-entering the kernel. Re-attach only when the full record is needed, and
    that read never waits.
    """
    register_path = run_root / "artifacts" / JOB_REGISTER
    spend_path = run_root / "artifacts" / "spend.jsonl"
    collected: list[dict[str, Any]] = []
    for notification in notifications:
        if notification.get("notification_type") != "compute_done":
            continue
        payload = notification.get("payload", {})
        job_id = payload.get("job_id")
        previous_jobs = load_jsonl(register_path) if register_path.is_file() else []
        # Take the submit row, not merely the newest row for this job. This loop
        # appends a completion row below, and a completion row carries no
        # run_fingerprint. A replayed notification would otherwise match its own
        # earlier completion, find no fingerprint, and raise on a job that was
        # already settled correctly.
        submitted = next(
            (
                row
                for row in reversed(previous_jobs)
                if row.get("job_id") == job_id and row.get("record") != "completion"
            ),
            {},
        )
        usage = payload.get("usage") or payload.get("settled_usage") or payload.get("billing")
        row = {
            "job_id": job_id,
            # Carry the attempt from the submit row. harvest_completed_receipt
            # refuses when the receipt's attempt_id does not equal this one, and
            # a completion row that omits it compares a real id against None and
            # can never pass. Absent a submit row this stays None, which keeps
            # that gate closed rather than opening it.
            "attempt_id": submitted.get("attempt_id"),
            "state": payload.get("state"),
            "exit_code": payload.get("exit_code"),
            "error_kind": payload.get("error_kind"),
            "system_hint": payload.get("system_hint"),
            "deadline_fired": payload.get("deadline_fired"),
            "output_files": payload.get("output_files"),
            "notes": payload.get("notes"),
            # Records whether this provider surface sent a per-job figure at all.
            # Without it the financial barrier cannot tell a surface that never
            # reports usage from one whose usage has not arrived yet, and it
            # waits forever on the first.
            "provider_usage_reported": usage is not None,
            "collected_at": utc_now(),
            "record": "completion",
        }
        append_jsonl(register_path, row)
        if usage is not None:
            if not isinstance(usage, dict):
                raise ValueError("a terminal completion usage record must be an object")
            amount = usage.get("amount")
            currency = usage.get("currency")
            amount_source = usage.get("amount_source") or usage.get("source")
            if isinstance(amount, bool) or not isinstance(amount, (int, float)) or amount < 0:
                raise ValueError("a terminal completion usage record must carry a nonnegative amount")
            if not isinstance(currency, str) or not currency:
                raise ValueError("a terminal completion usage record must carry a currency")
            if not isinstance(amount_source, str) or not amount_source:
                raise ValueError("a terminal completion usage record must name its source")
            run_fingerprint = payload.get("run_fingerprint") or submitted.get("run_fingerprint")
            if not isinstance(run_fingerprint, str) or not run_fingerprint:
                raise ValueError("a charge requires the submitted row's run fingerprint")
            with dispatch_lock(run_root):
                prior_charges = load_jsonl(spend_path) if spend_path.is_file() else []
                duplicate = any(
                    item.get("event") == "charge" and item.get("job_id") == job_id
                    for item in prior_charges
                )
                if not duplicate:
                    record_spend(
                        spend_path,
                        run_fingerprint=run_fingerprint,
                        amount=float(amount),
                        currency=currency,
                        amount_source=amount_source,
                        event="charge",
                        stage_id=submitted.get("stage_id"),
                        job_id=job_id,
                        provider_id=usage.get("provider_id") or payload.get("provider_id"),
                        details={key: value for key, value in usage.items() if key not in {"amount", "currency", "amount_source", "source"}},
                        note="settled provider usage from compute_done",
                    )
        collected.append(row)
    return collected


def _name_orphans(rows: Sequence[dict[str, Any]]) -> str:
    """Say which stage and attempt each orphan belongs to.

    An empty wave is closed without knowing which stage it was for, so the
    refusal has to carry that itself. Without it a stage that failed in local
    staging reports another stage's stranded reservation as its own problem.
    """
    return "; ".join(
        f"{row.get('stage_id')} attempt {row.get('attempt_id')} "
        f"submission {row.get('submission_id')}"
        for row in rows
    )


_ATTESTATION_TEXT_FIELDS = ("provider_id", "read_at", "read_by", "source")


def _checked_attestation(
    attestation: Any, *, subject_key: str, subject: str
) -> dict[str, Any]:
    """Validate the operator's read of the provider ledger, or refuse.

    This is the only thing separating a released reservation from a forgiven
    charge, so it has to say who read what, and it has to name the thing it was
    prepared for. An attestation that names nothing can be handed to a second
    orphan by accident, which turns one real read of the ledger into two
    releases. Blank strings pass a presence check and establish nothing, so they
    are refused here too.
    """
    if not isinstance(attestation, dict):
        raise ValueError("the provider ledger attestation must be an object")
    required = (*_ATTESTATION_TEXT_FIELDS, subject_key, "no_provider_job_exists")
    missing = [key for key in required if key not in attestation]
    if missing:
        raise ValueError(
            "the provider ledger attestation is missing " + ", ".join(missing)
        )
    for key in (*_ATTESTATION_TEXT_FIELDS, subject_key):
        value = attestation[key]
        if not isinstance(value, str) or not value.strip():
            raise ValueError(
                f"the provider ledger attestation field {key!r} must be a non-empty string"
            )
    if attestation[subject_key] != subject:
        raise ValueError(
            f"the provider ledger attestation names {subject_key} "
            f"{attestation[subject_key]!r}, not {subject!r}"
        )
    if attestation["no_provider_job_exists"] is not True:
        raise ValueError(
            "refusing to release a reservation the operator has not confirmed absent "
            "from the provider ledger"
        )
    return dict(attestation)


def orphaned_submissions(
    run_root: Path,
    *,
    run_fingerprint: str | None = None,
    stage_id: str | None = None,
    attempt_id: str | None = None,
) -> list[dict[str, Any]]:
    """Submissions that reserved budget and never became a provider job.

    ``submit_wave`` writes a ``submission-intent`` row before it calls
    ``submit_job``, and a ``submission-uncertain`` row when that call raises.
    Both carry a ``reservation_id`` and no ``job_id``, and ``merge_register``
    drops every row without a ``job_id``, so neither kind reaches any other
    reader. That is how a reservation goes missing: a staging path resolves
    against the wrong root, ``submit_job`` raises ENOENT before the provider
    creates anything, the wave comes back empty, and the attempt keeps a
    reservation that nothing in the register explains.

    A row here is neither a lost charge nor a free one. The dispatcher saw an
    exception, which does not say whether the provider created a job, so
    clearing one goes through ``void_submission`` and the operator's own read of
    the provider ledger.
    """
    register_path = run_root / "artifacts" / JOB_REGISTER
    rows = load_jsonl(register_path) if register_path.is_file() else []
    # Key on the whole identity, not the submission id alone. Resolving by id
    # across the register lets a row from another run, stage or attempt answer
    # for this one, and the answer it gives is that the reservation is fine.
    intents: dict[tuple, dict[str, Any]] = {}
    resolved: set[tuple] = set()
    for row in rows:
        raw_id = row.get("submission_id")
        if not raw_id:
            continue
        key = (
            str(raw_id),
            row.get("run_fingerprint"),
            row.get("stage_id"),
            row.get("attempt_id"),
        )
        record = row.get("record")
        if record in {"submission-intent", "submission-uncertain"}:
            intents.setdefault(key, {}).update(
                {name: value for name, value in row.items() if value is not None}
            )
            continue
        # A submission row carrying a job id is the provider's own answer that
        # the job exists, and a void row is the operator's that it does not.
        if record == "submission-void" or row.get("job_id"):
            resolved.add(key)
    orphans = [row for key, row in intents.items() if key not in resolved]
    if run_fingerprint is not None:
        orphans = [row for row in orphans if row.get("run_fingerprint") == run_fingerprint]
    if stage_id is not None:
        orphans = [row for row in orphans if row.get("stage_id") == stage_id]
    if attempt_id is not None:
        orphans = [row for row in orphans if row.get("attempt_id") == attempt_id]
    return orphans


def orphaned_reservations(run_root: Path) -> list[dict[str, Any]]:
    """Reservations holding budget with no submission and no job behind them.

    ``submit_wave`` records the reservation, then calls ``host.compute.create``,
    then writes the ``submission-intent`` row. Anything that fails in that
    interval strands the reservation where ``orphaned_submissions`` cannot see
    it, because no submission row was ever written, and the barrier then passes
    the empty wave for the same reason an earlier empty wave passed. A provider error from
    ``create`` reaches this on an ordinary error path, not only on a crash.

    A reservation counts as stranded only when its attempt appears nowhere in
    the register. One submission row of any kind means the attempt got far
    enough for ``orphaned_submissions`` to speak for it.
    """
    spend_path = run_root / "artifacts" / "spend.jsonl"
    spend_rows = load_jsonl(spend_path) if spend_path.is_file() else []
    register_path = run_root / "artifacts" / JOB_REGISTER
    register = load_jsonl(register_path) if register_path.is_file() else []
    known = {
        (row.get("run_fingerprint"), row.get("stage_id"), row.get("attempt_id"))
        for row in register
        if row.get("job_id") or row.get("submission_id")
    }
    stranded: list[dict[str, Any]] = []
    for row in spend_rows:
        details = row.get("details")
        if row.get("event") != "charge-estimate" or not isinstance(details, dict):
            continue
        if details.get("dispatch_reservation") is not True:
            continue
        key = (
            row.get("run_fingerprint"),
            row.get("stage_id"),
            details.get("attempt_id"),
        )
        if key in known:
            continue
        if _reservation_is_reconciled(spend_rows, str(details.get("reservation_id"))):
            continue
        stranded.append(row)
    return stranded


def void_reservation(
    run_root: Path,
    reservation_id: str,
    *,
    provider_ledger_attestation: dict[str, Any],
) -> dict[str, Any]:
    """Release a reservation that never reached a submission at all.

    The sibling of ``void_submission``, for the case where there is no
    submission id to name because none was ever written. Same gate: the operator
    reads the provider ledger, finds no job for this attempt, and says so. The
    attestation names the reservation rather than a submission.

    Refuses unless ``orphaned_reservations`` reports this reservation, so it
    cannot touch an attempt that has submissions or jobs of its own.
    """
    attestation = _checked_attestation(
        provider_ledger_attestation,
        subject_key="reservation_id",
        subject=str(reservation_id),
    )
    with dispatch_lock(run_root):
        row = next(
            (
                item
                for item in orphaned_reservations(run_root)
                if str(item["details"].get("reservation_id")) == str(reservation_id)
            ),
            None,
        )
        if row is None:
            raise ValueError(
                f"reservation {reservation_id} is not a stranded reservation"
            )
        run_fingerprint = str(row["run_fingerprint"])
        stage_id = str(row["stage_id"])
        attempt_id = str(row["details"]["attempt_id"])
        amount = float(row["amount"])
        spend_path = run_root / "artifacts" / "spend.jsonl"
        correction = record_spend(
            spend_path,
            run_fingerprint=run_fingerprint,
            amount=-amount,
            currency=row.get("currency"),
            amount_source="operator-attested absent provider job",
            event="charge",
            stage_id=stage_id,
            provider_id=str(attestation["provider_id"]),
            budget_maximum=row.get("budget_maximum"),
            details={
                "estimated": False,
                "reconciliation": True,
                "reconciled_estimate_amount": amount,
                "settled_amount": 0.0,
                "difference": -amount,
                "dispatch_reservation_reconciliation": True,
                "reservation_void": True,
                "reservation_id": str(reservation_id),
                "attempt_id": attempt_id,
                "provider_ledger_attestation": attestation,
            },
            note=(
                "voided a reservation that never reached a submission: the operator read "
                "the provider ledger and found no job for this attempt, so the "
                "reservation returns to the cap in full"
            ),
        )
        append_jsonl(
            run_root / "artifacts" / JOB_REGISTER,
            {
                "record": "reservation-void",
                "run_fingerprint": run_fingerprint,
                "stage_id": stage_id,
                "attempt_id": attempt_id,
                "reservation_id": str(reservation_id),
                "provider_ledger_attestation": attestation,
                "voided_at": utc_now(),
            },
        )
        return {
            "ok": True,
            "reservation_id": str(reservation_id),
            "reconciliation": correction,
            "released_estimate_amount": amount,
            "reason": "the stranded reservation is released",
        }


def _attempt_financial_barrier(
    run_root: Path,
    submitted: Sequence[dict[str, Any]],
    *,
    volume_barrier: bool,
) -> dict[str, Any]:
    """Reconcile a terminal attempt reservation after every job has settled."""
    # Read both registers under the lock. submit_wave holds it while it writes
    # the reservation and then the intent row, and reading between those two
    # appends is exactly how a stranded reservation looks like a clean run.
    with dispatch_lock(run_root):
        orphans = orphaned_submissions(run_root)
        stranded = orphaned_reservations(run_root) if not submitted else []
    if not submitted:
        if stranded:
            return {
                "ok": False,
                "terminal": True,
                "stranded_reservation_ids": [
                    str(row["details"].get("reservation_id")) for row in stranded
                ],
                "reason": (
                    f"{len(stranded)} reservation(s) hold budget with no submission and "
                    "no job behind them: "
                    + "; ".join(
                        f"{row.get('stage_id')} attempt "
                        f"{row['details'].get('attempt_id')}"
                        for row in stranded
                    )
                    + "; read the provider ledger and clear each one with "
                    "void_reservation"
                ),
            }
        # An empty wave used to pass here unconditionally, which lets a run
        # close clean while holding a reservation it never released. An empty
        # wave is exactly the shape a failed submit leaves behind, so refuse
        # while any submission still holds budget without a job.
        if orphans:
            return {
                "ok": False,
                "terminal": True,
                "orphaned_submission_ids": [
                    str(row.get("submission_id")) for row in orphans
                ],
                "reason": (
                    f"{len(orphans)} submission(s) reserved budget and never became a "
                    f"provider job: {_name_orphans(orphans)}; read the provider ledger "
                    "and clear each one with void_submission"
                ),
            }
        return {"ok": True, "terminal": False, "reason": "the wave has no jobs"}
    stage_id = str(submitted[0].get("stage_id"))
    attempt_id = str(submitted[0].get("attempt_id"))
    run_fingerprint = str(submitted[0].get("run_fingerprint"))
    register_path = run_root / "artifacts" / JOB_REGISTER
    merged = merge_register(load_jsonl(register_path) if register_path.is_file() else [])
    jobs = [
        row
        for row in merged
        if row.get("run_fingerprint") == run_fingerprint
        and row.get("stage_id") == stage_id
        and row.get("attempt_id") == attempt_id
    ]
    attempt_orphans = [
        row
        for row in orphans
        if row.get("run_fingerprint") == run_fingerprint
        and row.get("stage_id") == stage_id
        and row.get("attempt_id") == attempt_id
    ]
    if attempt_orphans:
        # A wave that submitted some specs and failed on another leaves both a
        # job and an orphan under one attempt id. Reconciling the reservation
        # from the jobs alone would close the attempt over the orphan.
        return {
            "ok": False,
            "terminal": True,
            "orphaned_submission_ids": [
                str(row.get("submission_id")) for row in attempt_orphans
            ],
            "reason": (
                f"{len(attempt_orphans)} submission(s) on this attempt reserved budget "
                f"and never became a provider job: {_name_orphans(attempt_orphans)}; "
                "read the provider ledger and clear each one with void_submission"
            ),
        }
    failed = any(
        row.get("state") not in {None, "submitted", "succeeded"}
        or (row.get("exit_code") is not None and row.get("exit_code") != 0)
        for row in jobs
    )
    terminal = failed or any(row.get("phase") in {"single", "finalize"} for row in jobs)
    if not terminal:
        return {
            "ok": True,
            "terminal": False,
            "reason": "the active reservation covers the next phase of this attempt",
        }
    if not volume_barrier:
        return {
            "ok": False,
            "terminal": True,
            "reason": "terminal attempt still has an open provider handle",
        }

    spend_path = run_root / "artifacts" / "spend.jsonl"
    with dispatch_lock(run_root):
        spend_rows = load_jsonl(spend_path) if spend_path.is_file() else []
        reservations = _reservation_rows(
            spend_rows,
            run_fingerprint=run_fingerprint,
            stage_id=stage_id,
            attempt_id=attempt_id,
        )
        if len(reservations) != 1:
            return {
                "ok": False,
                "terminal": True,
                "reason": f"expected one dispatch reservation, found {len(reservations)}",
            }
        reservation = reservations[0]
        reservation_id = str(reservation["details"]["reservation_id"])
        if _reservation_is_reconciled(spend_rows, reservation_id):
            return {
                "ok": True,
                "terminal": True,
                "reservation_id": reservation_id,
                "reason": "the dispatch reservation is already reconciled",
            }
        charged_job_ids = {
            row.get("job_id")
            for row in spend_rows
            if row.get("run_fingerprint") == run_fingerprint
            and row.get("event") == "charge"
            and row.get("job_id")
            and not (
                isinstance(row.get("details"), dict)
                and row["details"].get("reconciliation") is True
            )
        }
        unsettled = [row.get("job_id") for row in jobs if row.get("job_id") not in charged_job_ids]
        # A provider surface that never reports per-job usage would otherwise hold
        # this barrier open forever, which leaves the attempt unreconciled and so
        # makes every retry unauthorizable. Distinguish that from usage that has
        # simply not arrived yet: only a completion row that explicitly recorded
        # ``provider_usage_reported: False`` counts as reported-nothing. An older
        # row that predates the field says nothing either way and still waits.
        silent_job_ids = {
            str(row.get("job_id"))
            for row in load_jsonl(run_root / "artifacts" / JOB_REGISTER)
            if row.get("record") == "completion"
            and row.get("provider_usage_reported") is False
            and row.get("job_id")
        }
        settlement_unavailable = bool(unsettled) and all(
            str(job_id) in silent_job_ids for job_id in unsettled
        )
        if unsettled and not settlement_unavailable:
            return {
                "ok": False,
                "terminal": True,
                "reservation_id": reservation_id,
                "unsettled_job_ids": unsettled,
                "reason": "provider-settled usage is missing for one or more terminal jobs",
            }
        amount = float(reservation["amount"])
        if settlement_unavailable:
            # Close the attempt without giving its budget back.
            #
            # This row carries a zero amount and is deliberately not a
            # ``reconciliation`` row. ``spend_ledger_totals`` subtracts a
            # reconciliation row's ``reconciled_estimate_amount`` from the
            # estimated total, so writing one here would return this whole
            # reservation to ``enforce_spend_cap`` while the provider charge
            # behind it is still unknown. The jobs ran, so that charge is not
            # zero, and a later attempt could then spend past the operator's cap
            # without the ledger ever showing it. Leaving the original
            # ``charge-estimate`` row untouched keeps the approved maximum
            # standing against the cap, which is the conservative reading of a
            # charge nobody has measured.
            #
            # The marker still satisfies ``_reservation_is_reconciled``, so the
            # attempt closes and a retry becomes authorizable. That clears the
            # deadlock without claiming the attempt was free. Settling the real
            # figure needs provider billing evidence the operator attaches; until
            # then the cap sees the estimate.
            correction = record_spend(
                spend_path,
                run_fingerprint=run_fingerprint,
                amount=0.0,
                currency=reservation.get("currency"),
                amount_source="Modal attempt closed with no per-job usage reported",
                event="charge",
                stage_id=stage_id,
                provider_id="modal",
                budget_maximum=reservation.get("budget_maximum"),
                details={
                    "estimated": False,
                    "dispatch_reservation_reconciliation": True,
                    "settlement_unavailable": True,
                    "outstanding_estimate_amount": amount,
                    "reservation_id": reservation_id,
                    "attempt_id": attempt_id,
                    "unsettled_job_ids": [str(job_id) for job_id in unsettled],
                },
                note=(
                    "closed the attempt unsettled: this provider surface reported no "
                    "per-job usage, so the charge is unknown here. The reservation "
                    f"of {amount} stays outstanding against the budget cap until the "
                    "workspace billing report settles it."
                ),
            )
            return {
                "ok": True,
                "terminal": True,
                "reservation_id": reservation_id,
                "reconciliation": correction,
                "settlement_unavailable": True,
                "outstanding_estimate_amount": amount,
                "reason": (
                    "the terminal attempt is closed, and its unknown cost stays "
                    "outstanding against the cap"
                ),
            }
        correction = record_spend(
            spend_path,
            run_fingerprint=run_fingerprint,
            amount=-amount,
            currency=reservation.get("currency"),
            amount_source="automatic Modal attempt settlement",
            event="charge",
            stage_id=stage_id,
            provider_id="modal",
            budget_maximum=reservation.get("budget_maximum"),
            details={
                "estimated": False,
                "reconciliation": True,
                "reconciled_estimate_amount": amount,
                "settled_amount": 0.0,
                "difference": -amount,
                "dispatch_reservation_reconciliation": True,
                "reservation_id": reservation_id,
                "attempt_id": attempt_id,
            },
            note="cleared the attempt reservation after every provider job settled",
        )
        return {
            "ok": True,
            "terminal": True,
            "reservation_id": reservation_id,
            "reconciliation": correction,
            "settlement_unavailable": False,
            "reason": "the terminal attempt is closed and financially reconciled",
        }


def void_submission(
    run_root: Path,
    submission_id: str,
    *,
    provider_ledger_attestation: dict[str, Any],
) -> dict[str, Any]:
    """Release the reservation behind a submission the provider never created.

    Call this for a submission ``orphaned_submissions`` reports. It writes a
    terminal ``submission-void`` row, and returns the reserved estimate to the
    cap in full when the attempt holds no provider job at all. That is the whole
    of that earlier case: the reservation existed for submissions that never
    became jobs, so nothing is spending against it.

    An attempt that also holds a real job keeps its reservation. The void row is
    still written, so the barrier stops refusing, and ``close_wave`` reconciles
    the attempt through the settlement branch as it always did. Releasing here
    would both hand back money a running job is spending and consume the
    reconciliation marker that branch checks for.

    Full release is right only when no provider job exists, and nothing here can
    tell that on its own: the register records what the dispatcher saw, and what
    it saw was an exception.

    So the release is gated on ``provider_ledger_attestation``. The operator
    reads the provider's own ledger, finds no job for this submission, and says
    so. A local staging failure should cost a retry rather than the stage, and
    an unattested guess should cost neither, so a missing or negative
    attestation refuses and leaves the reservation standing against the cap.
    """
    attestation = _checked_attestation(
        provider_ledger_attestation,
        subject_key="submission_id",
        subject=str(submission_id),
    )

    with dispatch_lock(run_root):
        row = next(
            (
                item
                for item in orphaned_submissions(run_root)
                if str(item.get("submission_id")) == str(submission_id)
            ),
            None,
        )
        if row is None:
            raise ValueError(
                f"submission {submission_id} is not an open orphaned submission"
            )
        run_fingerprint = str(row["run_fingerprint"])
        stage_id = str(row["stage_id"])
        attempt_id = str(row["attempt_id"])
        register_path = run_root / "artifacts" / JOB_REGISTER
        spend_path = run_root / "artifacts" / "spend.jsonl"
        spend_rows = load_jsonl(spend_path) if spend_path.is_file() else []
        reservations = _reservation_rows(
            spend_rows,
            run_fingerprint=run_fingerprint,
            stage_id=stage_id,
            attempt_id=attempt_id,
        )
        if len(reservations) != 1:
            raise ValueError(
                f"expected one dispatch reservation for {stage_id}, found "
                f"{len(reservations)}"
            )
        reservation = reservations[0]
        reservation_id = str(reservation["details"]["reservation_id"])
        # The intent row recorded the reservation it dispatched against. If that
        # disagrees with the one this attempt holds, the lookup found someone
        # else's money and nothing below is safe to write.
        declared = row.get("reservation_id")
        if declared is not None and str(declared) != reservation_id:
            raise ValueError(
                f"submission {submission_id} names reservation {declared}, but its "
                f"attempt holds {reservation_id}"
            )
        void_row = {
            "record": "submission-void",
            "submission_id": str(submission_id),
            "run_fingerprint": run_fingerprint,
            "run_id": row.get("run_id"),
            "stage_id": stage_id,
            "adapter_id": row.get("adapter_id"),
            "phase": row.get("phase"),
            "attempt_id": attempt_id,
            "reservation_id": reservation_id,
            "submit_error": row.get("submit_error"),
            "provider_ledger_attestation": attestation,
            "voided_at": utc_now(),
        }
        # A reservation covers the attempt, not one submission. If any other
        # submission on this attempt became a provider job, that job is spending
        # against this reservation, and the settlement branch of
        # ``_attempt_financial_barrier`` is the only writer allowed to reconcile
        # it. Writing the marker here would satisfy ``_reservation_is_reconciled``
        # and send the barrier down its early return, closing the attempt without
        # ever checking that the job which really ran reported its usage.
        attempt_has_a_job = any(
            row.get("job_id")
            and row.get("run_fingerprint") == run_fingerprint
            and row.get("stage_id") == stage_id
            and row.get("attempt_id") == attempt_id
            for row in load_jsonl(register_path)
        )
        if attempt_has_a_job:
            append_jsonl(register_path, void_row)
            return {
                "ok": True,
                "submission_id": str(submission_id),
                "reservation_id": reservation_id,
                "reconciliation": None,
                "released_estimate_amount": 0.0,
                "reason": (
                    "the submission is void, and its attempt still holds provider jobs, so "
                    "the reservation stays with them until close_wave settles them"
                ),
            }
        if _reservation_is_reconciled(spend_rows, reservation_id):
            append_jsonl(register_path, void_row)
            return {
                "ok": True,
                "submission_id": str(submission_id),
                "reservation_id": reservation_id,
                "reconciliation": None,
                "released_estimate_amount": 0.0,
                "reason": "the submission is void and its reservation was already reconciled",
            }
        amount = float(reservation["amount"])
        correction = record_spend(
            spend_path,
            run_fingerprint=run_fingerprint,
            amount=-amount,
            currency=reservation.get("currency"),
            amount_source="operator-attested absent provider job",
            event="charge",
            stage_id=stage_id,
            provider_id=str(attestation["provider_id"]),
            budget_maximum=reservation.get("budget_maximum"),
            details={
                "estimated": False,
                "reconciliation": True,
                "reconciled_estimate_amount": amount,
                "settled_amount": 0.0,
                "difference": -amount,
                "dispatch_reservation_reconciliation": True,
                "submission_void": True,
                "reservation_id": reservation_id,
                "attempt_id": attempt_id,
                "submission_id": str(submission_id),
                "provider_ledger_attestation": attestation,
            },
            note=(
                "voided a submission the provider never created: the operator read the "
                "provider ledger and found no job for it, so the reservation returns to "
                "the cap in full"
            ),
        )
        append_jsonl(register_path, void_row)
        return {
            "ok": True,
            "submission_id": str(submission_id),
            "reservation_id": reservation_id,
            "reconciliation": correction,
            "released_estimate_amount": amount,
            "reason": "the submission is void and its reservation is released",
        }


def close_wave(
    submitted: Sequence[dict[str, Any]],
    collected: Sequence[dict[str, Any]],
    *,
    run_root: Path,
) -> dict[str, Any]:
    """Close every handle whose job has finished, and say whether the wave is done.

    This is the wave barrier. Closing is what ends a wave, so treat the
    returned ``barrier`` as the gate on submitting the next one. Do not move
    this call later in the loop, and do not drop it from a wave that only
    collects notifications. A reader who takes the close for tidiness will move
    it, and moving it breaks the commit guarantee below.

    Correctness is the first reason, and it is the part that is easy to miss. A
    job body here is bash with no Modal SDK, so there is no ``volume.commit()``
    to call, and the shell ``sync`` form is Volumes v2 only
    (volume_visibility.md:118-120). What a sandbox gets instead is background
    commits every few seconds while it executes, plus a final commit when it
    terminates (volume_visibility.md:125-127). The harvest does not terminate
    the sandbox (SKILL.md:466-469). So ``close()`` is the only guaranteed
    Volume commit point this surface offers us. A wave that skips it can leave
    the next wave reading a partial set of inputs, and that failure arrives
    looking like a science result rather than a mount one.

    Cost is the second reason. A finished job's container bills idle until the
    close, fifteen minutes of inactivity, or the container timeout, whichever
    comes first (SKILL.md:466-472). A fan-out creates one handle per shard, so
    an unclosed wave bills every shard for time it did not use.

    A terminal result or harvest failure still closes promptly. Missing output
    blocks scientific success and preserves the provider job/volume references
    for recovery, but never turns receipt safety into an unbounded billing leak.

    Every close is attempted even when an earlier one raises. A close that
    fails leaves a container billing and a commit unmade, so its failure is
    recorded and returned rather than allowed to abort the loop and strand the
    rest of the wave.

    Rows are marked closed in place, so calling this once per notification
    batch closes each handle exactly once. Pass the same ``submitted`` list
    every time.
    """
    register_path = run_root / "artifacts" / JOB_REGISTER
    outcome = {row["job_id"]: row for row in collected if row.get("job_id")}

    closed: list[str] = []
    held: list[dict[str, Any]] = []
    close_failed: list[dict[str, Any]] = []
    awaiting: list[str] = []
    terminal: list[dict[str, Any]] = []
    charges: list[dict[str, Any]] = []
    settlement_failed: list[dict[str, Any]] = []

    for row in submitted:
        if row.get("closed"):
            continue
        job_id = row.get("job_id")
        completion = outcome.get(job_id)
        if completion is None:
            # No compute_done for this job yet, so it may still be running.
            # Closing now would kill it and lose its unharvested output.
            awaiting.append(job_id)
            continue
        if row.get("recovered_from_provider_ledger") and row.get("job") is None:
            # A row recovered from the provider ledger carries neither the live
            # handle nor the result reader, because the process that submitted
            # the job is gone and neither object outlives it. Demanding them
            # left the job in ``still_open`` forever, behind a financial barrier
            # no later call could clear, and told the next frame to close a
            # handle that does not exist. The provider's own completion is the
            # terminal record here, so settle from it and treat the close as
            # already done: this process owns no sandbox to leak.
            details = {
                "job_id": job_id,
                "record": "terminal",
                "terminal_state": completion.get("state"),
                "exit_code": completion.get("exit_code"),
                "wall_s": completion.get("wall_s"),
                "notes": completion.get("notes"),
                "recovered_from_provider_ledger": True,
                "settled_at": utc_now(),
            }
            append_jsonl(register_path, details)
            terminal.append(details)
            try:
                # The completion payload is read exactly the way a live result
                # is, so a provider surface that does report usage still settles
                # a recovered job. One that reports none leaves the reservation
                # outstanding, which the attempt barrier already handles.
                with dispatch_lock(run_root):
                    charge = _record_result_charge(run_root, row, completion)
                if charge is not None:
                    charges.append(charge)
            except Exception as exc:  # one bad settle must not strand the wave
                settlement_failed.append(
                    {"job_id": job_id, "error": f"{type(exc).__name__}: {exc}"}
                )
            row["closed"] = True
            closed.append(job_id)
            append_jsonl(
                register_path,
                {
                    "job_id": job_id,
                    "record": "close",
                    "close_state": "closed",
                    "close_report": (
                        "recovered from the provider ledger: this process owns no "
                        "handle, and the provider reported the job terminal"
                    ),
                    "recovered_from_provider_ledger": True,
                    "closed_at": utc_now(),
                },
            )
            continue
        job = row.get("job")
        if job is None:
            settlement_failed.append(
                {"job_id": job_id, "error": "the row carries no job result reader"}
            )
        else:
            try:
                result = job.result()
                details = {
                    "job_id": job_id,
                    "record": "terminal",
                    "terminal_state": _result_value(
                        result, "state", completion.get("state")
                    ),
                    "exit_code": _result_value(
                        result, "exit_code", completion.get("exit_code")
                    ),
                    "wall_s": _result_value(result, "wall_s"),
                    "notes": _result_value(result, "notes"),
                    "settled_at": utc_now(),
                }
                append_jsonl(register_path, details)
                terminal.append(details)
                with dispatch_lock(run_root):
                    charge = _record_result_charge(run_root, row, result)
                if charge is not None:
                    charges.append(charge)
            except Exception as exc:
                detail = f"{type(exc).__name__}: {exc}"
                settlement_failed.append({"job_id": job_id, "error": detail})
                append_jsonl(
                    register_path,
                    {
                        "job_id": job_id,
                        "record": "terminal",
                        "terminal_result_error": detail,
                        "settled_at": utc_now(),
                    },
                )
        handle = row.get("handle")
        if handle is None:
            close_failed.append(
                {"job_id": job_id, "error": "the row carries no handle to close"}
            )
            continue
        try:
            report = handle.close(
                intent=(
                    f"shard finished, commit the Volume and stop the sandbox: "
                    f"{row.get('intent')}"
                )
            )
        except Exception as exc:  # one bad close must not strand the wave
            detail = f"{type(exc).__name__}: {exc}"
            close_failed.append({"job_id": job_id, "error": detail})
            append_jsonl(
                register_path,
                {
                    "job_id": job_id,
                    "record": "close",
                    "close_state": "failed",
                    "close_error": detail,
                    "close_seen_at": utc_now(),
                },
            )
            continue
        if isinstance(report, dict) and (
            report.get("ok") is False
            or report.get("state") in {"failed", "error", "refused"}
        ):
            detail = str(
                report.get("error")
                or report.get("reason")
                or "Modal handle close reported failure"
            )
            close_failed.append({"job_id": job_id, "error": detail})
            append_jsonl(
                register_path,
                {
                    "job_id": job_id,
                    "record": "close",
                    "close_state": "failed",
                    "close_error": detail,
                    "close_seen_at": utc_now(),
                },
            )
            continue
        row["closed"] = True
        closed.append(job_id)
        append_jsonl(
            register_path,
            {
                "job_id": job_id,
                "record": "close",
                "close_state": "closed",
                "close_report": str(report) if report is not None else None,
                "closed_at": utc_now(),
            },
        )

    still_open = [row.get("job_id") for row in submitted if not row.get("closed")]
    volume_barrier = not still_open
    financial = _attempt_financial_barrier(
        run_root, submitted, volume_barrier=volume_barrier
    )
    barrier = volume_barrier and financial["ok"]
    if barrier:
        next_action = (
            "every handle is closed and the attempt's financial barrier is met; "
            "the next authorized wave can proceed"
        )
    elif close_failed:
        next_action = (
            f"close failed on {len(close_failed)} handle(s), which are still "
            "billing and still uncommitted; call close_wave again before "
            "submitting anything"
        )
    elif volume_barrier and not financial["ok"]:
        next_action = (
            "the Volume is committed, but the financial barrier is closed: "
            f"{financial['reason']}"
        )
    else:
        next_action = (
            f"{len(awaiting)} job(s) have sent no compute_done; park on "
            "wait_for_notification, collect, and call close_wave again"
        )

    return {
        "closed": closed,
        "held": held,
        "close_failed": close_failed,
        "terminal": terminal,
        "charges": charges,
        "settlement_failed": settlement_failed,
        "awaiting_completion": awaiting,
        "still_open": still_open,
        "volume_barrier": volume_barrier,
        "financial_barrier": financial,
        "barrier": barrier,
        "next_action": next_action,
    }


def smoke_evidence(
    workspace: Path,
    collected: Sequence[dict[str, Any]],
    job_id: str,
    *,
    run_fingerprint: str,
    stage_id: str,
) -> dict[str, Any]:
    """What the host can prove about a smoke job from files it holds.

    An exit code proves the process ended. It does not prove the adapter scored
    anything. One completed run folded its one candidate, wrote its observation to the
    Volume, harvested zero files and exited 0, and the gate reading that exit
    code was ready to authorize the paid scale wave. Nine earlier runs hid it,
    because every failure path copies the receipt its ``--failure-kind`` call
    writes, so a failing smoke returned two files and a passing one returned
    none.

    So read the phase validation the container returns, bind it to this job's
    run, stage and attempt, and count what it declares. The stage is proven when
    the outputs are counted and at least one parse came back, which is the same
    bar the campaign applies everywhere else.
    """
    row = next((item for item in collected if item.get("job_id") == job_id), None)
    errors: list[str] = []
    result: dict[str, Any] = {
        "ok": False,
        "job_id": job_id,
        "stage_id": stage_id,
        "state": None if row is None else row.get("state"),
        "exit_code": None if row is None else row.get("exit_code"),
        "validation_path": None,
        "artifact_count": 0,
        "records": 0,
        "parser_ok": False,
        "errors": errors,
    }
    if row is None:
        errors.append(f"no collected row for job {job_id}")
        return result
    # A deadline lands as ``state == 'timed_out'`` rather than a generic failure,
    # with partial outputs already harvested. That is still a failed gate. Do not
    # submit scale shards for a stage whose smoke timed out; raise the timeout or
    # fix the adapter first.
    if row.get("state") != "succeeded" or row.get("exit_code") != 0:
        errors.append(
            f"smoke job {job_id} ended {row.get('state')!r} with exit code "
            f"{row.get('exit_code')!r}"
        )
        return result
    path = harvested_phase_validation_path(workspace, job_id, row.get("output_files"))
    result["validation_path"] = str(path)
    if not path.is_file():
        errors.append(f"harvested phase validation is missing: {path}")
        return result
    try:
        validation = load_json(path)
    except Exception as exc:
        errors.append(
            f"harvested phase validation is unreadable: {type(exc).__name__}: {exc}"
        )
        return result
    if not isinstance(validation, dict):
        errors.append("harvested phase validation is not an object")
        return result
    if validation.get("phase") != "smoke":
        errors.append(
            f"harvested phase validation names phase {validation.get('phase')!r}"
        )
    if validation.get("stage_id") != stage_id:
        errors.append("harvested phase validation belongs to another stage")
    if validation.get("run_fingerprint") != run_fingerprint:
        errors.append("harvested phase validation belongs to another run")
    attempt_id = row.get("attempt_id")
    if attempt_id is not None and validation.get("attempt_id") != attempt_id:
        errors.append("harvested phase validation belongs to another attempt")
    manifest = validation.get("manifest")
    artifacts = [
        item
        for item in ((manifest or {}).get("artifacts") or [])
        if isinstance(item, dict)
    ] if isinstance(manifest, dict) else []
    result["artifact_count"] = len(artifacts)
    result["records"] = sum(
        int(record.get("records") or 0)
        for artifact in artifacts
        for record in (artifact.get("files") or [])
        if isinstance(record, dict)
    )
    parsed = validation.get("parser_result")
    result["parser_ok"] = isinstance(parsed, dict) and parsed.get("ok") is True
    if validation.get("ok") is not True:
        declared = [str(error) for error in validation.get("errors") or []]
        errors.extend(declared or ["harvested phase validation reports ok false"])
    if not result["parser_ok"]:
        errors.append("the smoke phase parsed no result")
    if result["records"] < 1:
        errors.append("the smoke phase declared no output records")
    result["ok"] = not errors
    return result


def smoke_passed(
    collected: Sequence[dict[str, Any]],
    job_id: str,
    *,
    workspace: Path,
    run_fingerprint: str,
    stage_id: str,
) -> bool:
    """Whether the smoke job cleared the gate.

    This is the boolean form of ``smoke_evidence``. Call that one when a refusal
    has to say why. The keyword arguments are required on purpose: a gate that
    can be called without the evidence is a gate that reads the exit code.
    """
    return bool(
        smoke_evidence(
            workspace,
            collected,
            job_id,
            run_fingerprint=run_fingerprint,
            stage_id=stage_id,
        )["ok"]
    )


def unresolved_jobs(register: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Jobs the register saw submitted and never saw finish.

    These are the dangerous rows on a resume. A submitted job with no completion
    may still be running and still billing. Resubmitting it doubles the spend
    and can leave two containers writing the same Volume paths. Attach to the id
    instead. ``host.compute.ledger()`` is the cheap check for what is still live.
    """
    return [row for row in merge_register(register) if row.get("state") == "submitted"]


def _result_value(result: Any, key: str, default: Any = None) -> Any:
    if isinstance(result, dict):
        return result.get(key, default)
    return getattr(result, key, default)


def _result_usage(result: Any) -> dict[str, Any] | None:
    for key in ("usage", "settled_usage", "billing"):
        value = _result_value(result, key)
        if value is not None:
            if not isinstance(value, dict):
                raise ValueError(f"terminal result {key} must be an object")
            return value
    return None


def _record_result_charge(
    run_root: Path,
    submitted: dict[str, Any],
    result: Any,
) -> dict[str, Any] | None:
    usage = _result_usage(result)
    if usage is None:
        return None
    amount = usage.get("amount")
    currency = usage.get("currency")
    amount_source = usage.get("amount_source") or usage.get("source")
    if isinstance(amount, bool) or not isinstance(amount, (int, float)) or amount < 0:
        raise ValueError("a terminal result usage record must carry a nonnegative amount")
    if not isinstance(currency, str) or not currency:
        raise ValueError("a terminal result usage record must carry a currency")
    if not isinstance(amount_source, str) or not amount_source:
        raise ValueError("a terminal result usage record must name its source")
    spend_path = run_root / "artifacts" / "spend.jsonl"
    prior = load_jsonl(spend_path) if spend_path.is_file() else []
    if any(
        item.get("event") == "charge" and item.get("job_id") == submitted["job_id"]
        for item in prior
    ):
        return None
    return record_spend(
        spend_path,
        run_fingerprint=submitted["run_fingerprint"],
        amount=float(amount),
        currency=currency,
        amount_source=amount_source,
        event="charge",
        stage_id=submitted.get("stage_id"),
        job_id=submitted["job_id"],
        provider_id=usage.get("provider_id"),
        details={
            key: value
            for key, value in usage.items()
            if key not in {"amount", "currency", "amount_source", "source"}
        },
        note="settled provider usage from reattached terminal result",
    )


def attach_pending_jobs(
    host: Any,
    plan: dict[str, Any],
    *,
    run_root: Path,
) -> dict[str, Any]:
    """Attach every submitted row left live when the previous kernel ended."""
    register_path = run_root / "artifacts" / JOB_REGISTER
    register = load_jsonl(register_path) if register_path.is_file() else []
    pending = unresolved_jobs(register)
    attached: list[dict[str, Any]] = []
    for row in pending:
        params = row.get("provider_params")
        if not isinstance(params, dict):
            raise ValueError(f"pending job {row.get('job_id')} has no provider parameters")
        handle = host.compute.create("modal", provider_params=params)
        job = handle.attach_job(str(row["job_id"]))
        attached.append(dict(row, handle=handle, job=job, reattached=True))
    return {
        "run_id": plan["run_id"],
        "pending": pending,
        "attached": attached,
        "next_action": (
            "call wait_for_notification, then settle_reattached_jobs"
            if attached
            else "no submitted job needs reattachment"
        ),
    }


def settle_reattached_jobs(
    attached: Sequence[dict[str, Any]],
    notifications: Iterable[dict[str, Any]],
    *,
    run_root: Path,
) -> dict[str, Any]:
    """Read terminal results, reconcile spend, and close reattached handles."""
    collected = collect_notifications(run_root, notifications)
    close = close_wave(attached, collected, run_root=run_root)
    return {
        "collected": collected,
        "terminal": close["terminal"],
        "charges": close["charges"],
        "close": close,
        "barrier": close["barrier"],
    }


def reattach_pending_jobs(
    host: Any,
    plan: dict[str, Any],
    *,
    run_root: Path,
    wait_for_notification: Any,
) -> dict[str, Any]:
    """Attach, wait once, settle terminal details, and close every handle."""
    attached = attach_pending_jobs(host, plan, run_root=run_root)
    if not attached["attached"]:
        return attached
    wake = wait_for_notification()
    if not isinstance(wake, dict) or wake.get("status") == "error":
        return {
            **attached,
            "ok": False,
            "errors": ["wait_for_notification returned no terminal notification"],
        }
    notifications = wake.get("notifications", [])
    settled = settle_reattached_jobs(
        attached["attached"], notifications, run_root=run_root
    )
    return {**attached, **settled, "ok": settled["barrier"]}


def modal_stage_ids(plan: dict[str, Any]) -> list[str]:
    """Return the stages owned by this dispatcher, in plan order.

    Materialized plans carry the authoritative paid-stage provider resolution.
    Older plans predate that table, so retain a compatibility fallback based on
    the Modal environment identity of the stage's adapter.  Local stages remain
    dependencies in the run plan, but they are neither invalid Modal adapters
    nor pending work for this dispatcher.
    """
    resolved = plan.get("paid_stage_providers")
    selected: set[str] = set()
    if isinstance(resolved, list):
        for row in resolved:
            if not isinstance(row, dict):
                continue
            providers = row.get("provider_ids")
            provider_ids = (
                {str(value) for value in providers}
                if isinstance(providers, list)
                else {str(row.get("provider_id", ""))}
            )
            if "modal" in provider_ids and row.get("stage_id"):
                selected.add(str(row["stage_id"]))
    else:
        adapters = {
            str(adapter.get("adapter_id")): adapter
            for adapter in plan.get("adapters", [])
            if isinstance(adapter, dict)
        }
        for stage in plan.get("stages", []):
            if not isinstance(stage, dict):
                continue
            adapter = adapters.get(str(stage.get("adapter_id")), {})
            if str(adapter.get("environment_identity", "")).startswith("modal-env:"):
                selected.add(str(stage.get("stage_id")))
    return [
        str(stage["stage_id"])
        for stage in plan.get("stages", [])
        if isinstance(stage, dict) and str(stage.get("stage_id")) in selected
    ]


def resume_report(
    plan: dict[str, Any],
    *,
    workspace: Path,
    run_root: Path,
) -> dict[str, Any]:
    """Say what a resumed run should do next, without submitting anything.

    Three categories come out. ``complete`` stages have a harvested receipt
    whose run fingerprint matches the plan. ``in_flight`` stages have a
    submitted job with no completion, and need attaching rather than
    resubmitting. Everything else is ``pending``.

    Fan-out widths are deliberately not read from the register here. A resumed
    run recomputes them from the upstream receipts, because the recorded number
    describes the run that wrote it and an upstream stage may have been rerun.
    """
    register_path = run_root / "artifacts" / JOB_REGISTER
    register = load_jsonl(register_path) if register_path.is_file() else []
    bound_submission_ids = {
        row.get("submission_id")
        for row in register
        if row.get("job_id") and row.get("submission_id")
    }
    ambiguous_submissions = [
        row
        for row in register
        if row.get("record") in {"submission-intent", "submission-uncertain"}
        and row.get("submission_id") not in bound_submission_ids
    ]

    owned_stage_ids = modal_stage_ids(plan)
    owned = set(owned_stage_ids)
    complete: list[str] = []
    pending: list[str] = []
    for stage in plan["stages"]:
        if stage["stage_id"] not in owned:
            continue
        receipt = find_stage_receipt(workspace, stage["stage_id"], register)
        if receipt and receipt.get("run_fingerprint") == plan["run_fingerprint"]:
            complete.append(stage["stage_id"])
        else:
            pending.append(stage["stage_id"])

    in_flight = unresolved_jobs(register)
    return {
        "run_id": plan["run_id"],
        "run_fingerprint": plan["run_fingerprint"],
        "complete": complete,
        "pending": pending,
        "ignored_non_modal_stages": [
            stage["stage_id"]
            for stage in plan["stages"]
            if stage["stage_id"] not in owned
        ],
        "in_flight": in_flight,
        "ambiguous_submissions": ambiguous_submissions,
        "next_action": (
            "inspect the provider ledger and bind each ambiguous submission before continuing"
            if ambiguous_submissions
            else "attach to the in-flight job ids before submitting anything else"
            if in_flight
            else f"plan the first pending stage: {pending[0]}" if pending
            else "every stage has a matching receipt"
        ),
    }


# --- verification, offline ---------------------------------------------------


def verify_plan(plan: dict[str, Any], policy: DispatchPolicy) -> dict[str, Any]:
    """Check every Modal-selected adapter can become a job, without touching Modal.

    This is free and catches the failures that are otherwise slow: an
    unresolved environment identity, a missing image reference, a GPU request no
    shipped tier satisfies, a command template still reading ``__REQUIRED__``.
    """
    errors: list[str] = []
    resolved: list[dict[str, Any]] = []
    owned_stage_ids = set(modal_stage_ids(plan))
    stages = [stage for stage in plan["stages"] if stage["stage_id"] in owned_stage_ids]
    selected_adapter_ids = {str(stage["adapter_id"]) for stage in stages}
    for adapter in plan["adapters"]:
        adapter_id = adapter.get("adapter_id", "<unnamed>")
        if str(adapter_id) not in selected_adapter_ids:
            continue
        try:
            volumes = volumes_for_adapter(adapter, policy)
            params = provider_params(
                adapter,
                volumes=volumes,
                container_timeout_s=policy.container_timeout_s,
            )
        except (ValueError, KeyError) as exc:
            errors.append(f"{adapter_id}: {exc}")
            continue
        if policy.python_executable and UNRESOLVED in policy.python_executable:
            errors.append(f"{adapter_id}: python_executable is unresolved")
        for key in ("command_argv_template", "parser_argv_template", "toolcheck_argv"):
            if any(UNRESOLVED in str(token) for token in adapter.get(key, [])):
                errors.append(f"{adapter_id}: {key} is still unresolved")
            if any("{{python_executable}}" in str(token) for token in adapter.get(key, [])) and not policy.python_executable:
                errors.append(f"{adapter_id}: python_executable is unresolved")
        resolved.append({"adapter_id": adapter_id, "provider_params": params})

    try:
        waves = stage_waves(plan)
    except ValueError as exc:
        errors.append(str(exc))
        waves = []

    fanning = [
        stage["stage_id"]
        for stage in stages
        if stage.get("mode") == "smoke_scale"
        and (stage.get("fanout") or {}).get("count_from") is not None
    ]

    # An adapter with no measured per-candidate wall clock falls back to the
    # stage's whole timeout. That never truncates real work and it costs the
    # full stage budget on a hang, so name every adapter still in that state.
    uncalibrated = sorted(
        item["adapter_id"]
        for item in resolved
        if item["adapter_id"] not in policy.seconds_per_candidate
    )

    # run_timeout_s above the container's remaining life is not a guard. The
    # harvest watchdog stops the job first, and the run lands as timed_out with
    # a deadline that was never the one that was set.
    over_container = []
    if policy.container_timeout_s is not None:
        over_container = sorted(
            stage["stage_id"]
            for stage in stages
            if int(stage["timeout_minutes"]) * 60 > policy.container_timeout_s
        )

    return {
        "ok": not errors,
        "modal_stage_ids": [stage["stage_id"] for stage in stages],
        "ignored_non_modal_stage_ids": [
            stage["stage_id"]
            for stage in plan["stages"]
            if stage["stage_id"] not in owned_stage_ids
        ],
        "adapters": resolved,
        "stage_wave_count": len(waves),
        "stage_waves": waves,
        "stages_resolving_width_at_run_time": fanning,
        "adapters_without_measured_seconds_per_candidate": uncalibrated,
        "stages_whose_timeout_exceeds_the_container": over_container,
        "errors": errors,
        "note": repo_bootstrap_note(policy),
    }


def dry_run_report(
    plan: dict[str, Any],
    *,
    stage_id: str,
    policy: DispatchPolicy,
    bundle_root: Path,
    source_repo: Path,
    receipts_dir: Path | None = None,
    register: Sequence[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Print the exact bootstrap transfer list without creating a handle.

    This is the free half of the paid path, so a refusal is reported rather than
    raised. Reading why the transfer list cannot be built costs nothing, and the
    same refusal reaches the dispatch path as an exception.
    """
    stage = next(item for item in plan["stages"] if item["stage_id"] == stage_id)
    adapter = next(item for item in plan["adapters"] if item["adapter_id"] == stage["adapter_id"])
    try:
        manifest = bootstrap_file_manifest(
            plan,
            stage,
            adapter,
            bundle_root=bundle_root,
            source_repo=source_repo,
            policy=policy,
            receipts_dir=receipts_dir,
            register=register,
        )
    except (FileNotFoundError, ValueError) as error:
        return {
            "ok": False,
            "stage_id": stage_id,
            "adapter_id": adapter["adapter_id"],
            "stage_inputs": list(stage.get("inputs", [])),
            "error": str(error),
        }
    return {
        "ok": True,
        "stage_id": stage_id,
        "adapter_id": adapter["adapter_id"],
        "stage_inputs": manifest["stage_inputs"],
        "stage_input_files": manifest["stage_input_files"],
        "stage_inputs_not_shipped": manifest["stage_inputs_not_shipped"],
        "stage_input_provenance": manifest["stage_input_provenance"],
        "stage_contract": manifest["stage_contract"],
        "volume_mappings": manifest["volume_mappings"],
        "transfer_count": len(manifest["files"]),
        "transfer_list": manifest["files"],
        "bootstrap": {
            "inputs": [BOOTSTRAP_SCRIPT, BOOTSTRAP_ARCHIVE, BOOTSTRAP_MANIFEST],
            "destination": f"{policy.volume_mount}:{policy.repo_dir} plus the plan runtime paths",
        },
    }


# --- command line ------------------------------------------------------------


def cli(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Plan Modal jobs for a binder-lane run. This command never submits. "
            "Submitting needs the host object, which exists only in the Claude "
            "Science repl kernel."
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    verify = subparsers.add_parser(
        "verify", help="check every adapter can become a Modal job"
    )
    verify.add_argument("--plan", type=Path, required=True)
    verify.add_argument("--policy", type=Path)

    dry_run = subparsers.add_parser(
        "dry-run", help="print the contract-driven bootstrap transfer list"
    )
    dry_run.add_argument("--plan", type=Path, required=True)
    dry_run.add_argument("--policy", type=Path, required=True)
    dry_run.add_argument("--stage", required=True)
    dry_run.add_argument("--bundle-root", type=Path)
    dry_run.add_argument("--source-repo", type=Path)

    shards = subparsers.add_parser(
        "shards", help="show how a width splits into shards for one adapter"
    )
    shards.add_argument("--adapter", required=True)
    shards.add_argument("--width", type=int, required=True)
    shards.add_argument("--policy", type=Path)

    waves = subparsers.add_parser("waves", help="show the stage dependency waves")
    waves.add_argument("--plan", type=Path, required=True)

    resume = subparsers.add_parser(
        "resume", help="say what a resumed run should do next"
    )
    resume.add_argument("--plan", type=Path, required=True)
    resume.add_argument("--workspace", type=Path, required=True)
    resume.add_argument("--run-root", type=Path, required=True)

    retry = subparsers.add_parser(
        "authorize-retry",
        help="append an operator authorization for a new paid attempt; never submits",
    )
    retry.add_argument("--plan", type=Path, required=True)
    retry.add_argument("--run-root", type=Path, required=True)
    retry.add_argument("--stage", required=True)
    retry.add_argument("--prior-attempt", required=True)
    retry.add_argument("--new-attempt", required=True)
    retry.add_argument("--authorized-by", required=True)
    retry.add_argument("--reason", required=True)

    bind = subparsers.add_parser(
        "bind-submission",
        help="bind a provider-ledger job ID to an uncertain submit; never submits",
    )
    bind.add_argument("--plan", type=Path, required=True)
    bind.add_argument("--run-root", type=Path, required=True)
    bind.add_argument("--submission-id", required=True)
    bind.add_argument("--job-id", required=True)

    args = parser.parse_args(argv)
    try:
        if args.command == "verify":
            result = verify_plan(load_json(args.plan), DispatchPolicy.load(args.policy))
        elif args.command == "dry-run":
            plan = load_json(args.plan)
            policy = DispatchPolicy.load(args.policy)
            source_repo = args.source_repo or Path(__file__).resolve().parents[3]
            result = dry_run_report(
                plan,
                stage_id=args.stage,
                policy=policy,
                bundle_root=(args.bundle_root or args.plan.parent).resolve(),
                source_repo=source_repo.resolve(),
            )
        elif args.command == "shards":
            policy = DispatchPolicy.load(args.policy)
            slices = split_shards(args.width, policy.width_for(args.adapter))
            result = {
                "ok": True,
                "adapter_id": args.adapter,
                "width": args.width,
                "shard_width": policy.width_for(args.adapter),
                "shards": [item.as_dict() for item in slices],
            }
        elif args.command == "waves":
            plan = load_json(args.plan)
            result = {"ok": True, "waves": stage_waves(plan)}
        elif args.command == "authorize-retry":
            row = authorize_retry(
                args.run_root,
                plan=load_json(args.plan),
                stage_id=args.stage,
                prior_attempt_id=args.prior_attempt,
                new_attempt_id=args.new_attempt,
                authorized_by=args.authorized_by,
                reason=args.reason,
            )
            result = {"ok": True, "authorization": row}
        elif args.command == "bind-submission":
            row = bind_uncertain_submission(
                args.run_root,
                plan=load_json(args.plan),
                submission_id=args.submission_id,
                job_id=args.job_id,
            )
            result = {"ok": True, "submission": row}
        else:
            result = dict(
                resume_report(
                    load_json(args.plan),
                    workspace=args.workspace,
                    run_root=args.run_root,
                ),
                ok=True,
            )
    except Exception as exc:  # surface the reason, never a traceback
        result = {"ok": False, "errors": [f"{type(exc).__name__}: {exc}"]}

    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(cli())
