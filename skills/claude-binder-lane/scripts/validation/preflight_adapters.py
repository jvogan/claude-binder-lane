#!/usr/bin/env python3
"""Run every cofold adapter end to end on a local fixture, for free.

The published three modes and the reducer have never executed against a real dispatcher
invocation. The failures this catches are the cheap ones: a sibling import that
breaks under PYTHONSAFEPATH, an argument the profile does not pass, a path built
wrong, a measurement that misses one of the 48 fields, a job that aborts on
candidate 2 instead of writing a failed row and continuing.

What this does and does not stand in for. The model forward pass is replaced.
Nothing else is. Each adapter's own entry script runs, parses its own arguments,
reads the campaign configuration, builds the plan, writes its artifacts through
binder_contract, computes every metric through binder_metrics, and handles its
own failures. For the two ESMFold2 arms the stand-in is a fake torch, a fake
transformers checkpoint loader and a fake esm package installed into sys.modules
by a sitecustomize file. For Protenix v2 the stand-in is a fake `protenix`
console script on PATH, so the arm's real subprocess call, its real output-tree
walk and its real file copying all run.

Every command is built from `command_argv_template` in a profile, so the argv is
the one the dispatcher renders rather than one written here. Two variants run per
arm. The bare variant is the template alone, which is what a paid run executes
today. The supplemented variant adds the per-target arguments the arm's own
parser declares and the template carries no token for, which is the only way to
reach the fold and the write.

Where the fixture's values come from:

- The target block, the site block and the binder chain letters are the shipped
  `claude_binder/data/templates/campaign.template.json`, with its
  `__REQUIRED__` placeholders filled in locally.
- `site_metric_basis` is one of the two values
  `claude_binder/data/schemas/cofold-observation.schema.json` registers.
- `site.hotspot_source` is `explicit`, the first of the names HOTSPOT_SOURCES
  registers in `claude_binder/lane.py`, and the
  hotspot list beside it is the same one the argv passes.
- Every `model_revision` is the value the profile carries for that adapter,
  including `__REQUIRED__` where the profile has not resolved one.
- The 48 measurement field names are read out of MEASUREMENT_SOURCE_FIELDS in
  `claude_binder/lane.py` at run time.
- Every hash in the fixture is computed from the file it names.

Exit code is zero when every adapter passes.

Usage:

    PYTHONSAFEPATH=1 python3 scripts/validation/preflight_adapters.py
    PYTHONSAFEPATH=1 python3 scripts/validation/preflight_adapters.py --work-dir DIR
"""

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import uuid
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]


def resolve_package_root() -> Path:
    """Return the package beside a built skill or inside the source checkout."""
    candidates = (
        REPO_ROOT / "claude_binder",
        REPO_ROOT.parents[1] / "src" / "claude_binder",
    )
    for candidate in candidates:
        if (candidate / "__init__.py").is_file():
            return candidate.resolve()
    raise SystemExit(
        "claude_binder package not found. Reinstall this skill; it does not "
        "ship the package these scripts import.")


PACKAGE_ROOT = resolve_package_root()
if str(PACKAGE_ROOT.parent) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT.parent))

from claude_binder import lane as EXECUTOR

EXECUTOR_PATH = PACKAGE_ROOT / "lane.py"
TEMPLATE_DIR = PACKAGE_ROOT / "data" / "templates"
CAMPAIGN_TEMPLATE_PATH = TEMPLATE_DIR / "campaign.template.json"
OBSERVATION_SCHEMA_PATH = PACKAGE_ROOT / "data" / "schemas" / "cofold-observation.schema.json"
DEFAULT_PROFILE_PATH = TEMPLATE_DIR / "profiles" / "modal-draft.json"


def load_executor() -> Any:
    """Load the executor module so token rendering follows the paid path."""
    return EXECUTOR


EXECUTOR = load_executor()

# The run phase names the directory under the attempt directory. `scale` is what
# the executor passes for the second phase of a smoke_scale stage, and it is the
# phase whose count is the resolved fanout rather than 1.
RUN_PHASE = "scale"
SCORER_RUN_PHASE = "single"

# Three candidates, and the middle one is the one the stand-in fails on. A
# failure on the last candidate would not show that the arm carried on.
CANDIDATE_BINDER_LENGTHS = (5, 6, 7)
FAILING_CANDIDATE_INDEX = 1
TARGET_LENGTH = 8

# The placeholder a profile carries where a value has not been resolved yet.
UNRESOLVED = "__REQUIRED__"

# --- the stand-in for the model, written to disk and loaded at interpreter start


FIXTURE_MODULE_SOURCE = '''
# Shared geometry for the preflight fixture. The fake ESMFold2 package and the
# fake protenix console script both build their structures here, so a prediction
# and the design pose it is scored against use one generator.
#
# The layout copies _synthetic_cif in binder_metrics.py: a full N, CA, C, O
# backbone per residue, the second chain shifted 6 Angstroms along x, residue n
# at y = n * 3.8. That places every same-numbered residue pair 6 Angstroms apart
# at the C-alpha and 4 Angstroms apart at the closest heavy atoms, which is
# inside the contact cutoffs and outside the clash cutoff.

BACKBONE_OFFSETS = (
    ("N", 0.0, 0.0, 0.0),
    ("CA", 1.0, 0.0, 0.0),
    ("C", 2.0, 0.0, 0.0),
    ("O", 2.0, 1.0, 0.0),
)

CIF_HEADER = (
    "data_preflight",
    "loop_",
    "_atom_site.group_PDB",
    "_atom_site.id",
    "_atom_site.type_symbol",
    "_atom_site.label_atom_id",
    "_atom_site.label_comp_id",
    "_atom_site.label_asym_id",
    "_atom_site.label_seq_id",
    "_atom_site.pdbx_PDB_ins_code",
    "_atom_site.Cartn_x",
    "_atom_site.Cartn_y",
    "_atom_site.Cartn_z",
    "_atom_site.occupancy",
    "_atom_site.B_iso_or_equiv",
    "_atom_site.auth_seq_id",
    "_atom_site.auth_asym_id",
    "_atom_site.pdbx_PDB_model_num",
)

# Target residues are alanine and binder residues are glycine. The fixture's
# known target and binder sequences therefore exercise sequence-based mapping.
B_FACTOR = 85.00


def complex_cif(chains):
    # chains is an ordered sequence of (chain_id, residue_count). The first
    # chain sits at x 0 and every later one is 6 Angstroms further along x.
    rows = []
    serial = 1
    for position, (chain_id, count) in enumerate(chains):
        shift = 6.0 * position
        code = "ALA" if position == 0 else "GLY"
        for number in range(1, count + 1):
            for name, dx, dy, dz in BACKBONE_OFFSETS:
                rows.append(
                    "ATOM {serial} {element} {name} {code} {chain} {number} . "
                    "{x:.3f} {y:.3f} {z:.3f} 1.00 {b:.2f} {number} {chain} 1".format(
                        serial=serial,
                        element=name[0],
                        name=name,
                        code=code,
                        chain=chain_id,
                        number=number,
                        x=shift + dx,
                        y=number * 3.8 + dy,
                        z=dz,
                        b=B_FACTOR,
                    )
                )
                serial += 1
    return "\\n".join(list(CIF_HEADER) + rows) + "\\n#\\n"


def pae_rows(size):
    # Deterministic and asymmetric enough that both ipSAE directions are
    # non-zero. Everything is under the 10 Angstrom interface cutoff.
    return [
        [0.5 if i == j else (1.0 if i < j else 4.25) for j in range(size)]
        for i in range(size)
    ]
'''


SITECUSTOMIZE_SOURCE = '''
# Installs the stand-in for the model into sys.modules before any adapter runs.
# The adapter's heavy imports are lazy and sit inside load_model and fold_one, so
# a module registered here is the one they find. Nothing in the adapter is
# stubbed: it still loads a checkpoint, still folds, still reads back a PAE and a
# complex, and still writes and measures its own artifacts.

import json
import os
import sys
import types

import numpy

import preflight_fixture


def _log(record):
    path = os.environ.get("PREFLIGHT_FAKE_LOG")
    if not path:
        return
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\\n")


class _PredictionFailed(RuntimeError):
    pass


# --- fake torch, for install_safe_svd -------------------------------------


def _install_torch():
    torch = types.ModuleType("torch")
    linalg = types.ModuleType("torch.linalg")

    def svd(A, full_matrices=True, driver=None):
        raise NotImplementedError("the preflight never reaches a real SVD")

    linalg.svd = svd
    torch.linalg = linalg
    torch.isfinite = numpy.isfinite
    torch.nan_to_num = numpy.nan_to_num
    sys.modules["torch"] = torch
    sys.modules["torch.linalg"] = linalg


# --- fake transformers checkpoint loader ----------------------------------


class _FakeModel:
    def __init__(self, checkpoint):
        self.checkpoint = checkpoint
        self.kernel_backend = None
        self.chunk_size = None
        self.device = None

    @classmethod
    def from_pretrained(cls, checkpoint):
        _log({"call": "from_pretrained", "checkpoint": checkpoint})
        return cls(checkpoint)

    def to(self, device):
        self.device = device
        return self

    def eval(self):
        return self

    def set_kernel_backend(self, name):
        self.kernel_backend = name

    def set_chunk_size(self, size):
        self.chunk_size = size


def _install_transformers():
    for name in ("transformers", "transformers.models", "transformers.models.esmfold2"):
        if name not in sys.modules:
            sys.modules[name] = types.ModuleType(name)
    modeling = types.ModuleType("transformers.models.esmfold2.modeling_esmfold2")
    modeling.ESMFold2Model = _FakeModel
    sys.modules["transformers.models.esmfold2.modeling_esmfold2"] = modeling
    sys.modules["transformers.models.esmfold2"].modeling_esmfold2 = modeling
    sys.modules["transformers.models"].esmfold2 = sys.modules["transformers.models.esmfold2"]
    sys.modules["transformers"].models = sys.modules["transformers.models"]


# --- fake esm package ------------------------------------------------------


class _ProteinInput:
    def __init__(self, id, sequence, msa=None):
        self.id = id
        self.sequence = sequence
        self.msa = msa


class _StructurePredictionInput:
    def __init__(self, sequences):
        self.sequences = sequences


class _Complex:
    def __init__(self, text):
        self._text = text

    def to_mmcif(self):
        return self._text


# `ESMFold2InputBuilder.fold` returns one MolecularComplexResult at one
# diffusion sample and a list otherwise. Source: local return-shape fixture.
class _MolecularComplexResult:
    pass


class _Prediction(_MolecularComplexResult):
    def __init__(self, chains):
        size = sum(count for _, count in chains)
        self.complex = _Complex(preflight_fixture.complex_cif(chains))
        # A tensor in the real arm. pae_matrix calls .tolist() and rounds, so the
        # stand-in has to carry the same method rather than a plain list.
        self.pae = numpy.array(preflight_fixture.pae_rows(size), dtype=float)
        self.plddt = numpy.full(size, preflight_fixture.B_FACTOR, dtype=float)
        self.iptm = 0.63
        self.ptm = 0.71


class _InputBuilder:
    def fold(self, model, spi, num_loops, num_sampling_steps, num_diffusion_samples, seed):
        chains = [(item.id, len(item.sequence)) for item in spi.sequences]
        _log(
            {
                "call": "fold",
                "checkpoint": model.checkpoint,
                "device": model.device,
                "kernel_backend": model.kernel_backend,
                "chunk_size": model.chunk_size,
                "num_loops": num_loops,
                "num_sampling_steps": num_sampling_steps,
                "num_diffusion_samples": num_diffusion_samples,
                "seed": seed,
                "chains": chains,
                "msa_present": [item.msa is not None for item in spi.sequences],
            }
        )
        failing = os.environ.get("PREFLIGHT_FAIL_BINDER_SEQUENCE")
        if failing and any(item.sequence == failing for item in spi.sequences):
            # The point of the whole exercise. A job that raises here must write a
            # failed row and go on to the next candidate.
            raise _PredictionFailed(
                "the preflight stand-in refused this candidate on purpose"
            )
        # Match the local fixture exactly at the one-sample boundary.
        predictions = [_Prediction(chains) for _ in range(num_diffusion_samples)]
        return predictions[0] if num_diffusion_samples == 1 else predictions


class _MSA:
    def __init__(self, path, max_sequences):
        self.path = path
        self.max_sequences = max_sequences

    @classmethod
    def from_a3m(cls, path, max_sequences=None):
        _log({"call": "from_a3m", "path": path, "max_sequences": max_sequences})
        return cls(path, max_sequences)


def _install_esm():
    for name in ("esm", "esm.models", "esm.utils", "esm.utils.msa"):
        if name not in sys.modules:
            sys.modules[name] = types.ModuleType(name)
    # `fold` returns this class at one sample. Source: local return-shape fixture.
    # The adapter imports it to check the documented return type.
    structure_module = types.ModuleType("esm.utils.structure")
    molecular_complex = types.ModuleType("esm.utils.structure.molecular_complex")
    molecular_complex.MolecularComplexResult = _MolecularComplexResult
    structure_module.molecular_complex = molecular_complex
    esmfold2 = types.ModuleType("esm.models.esmfold2")
    esmfold2.ESMFold2InputBuilder = _InputBuilder
    esmfold2.ProteinInput = _ProteinInput
    esmfold2.StructurePredictionInput = _StructurePredictionInput
    sys.modules["esm.models.esmfold2"] = esmfold2
    msa_module = types.ModuleType("esm.utils.msa.msa")
    msa_module.MSA = _MSA
    sys.modules["esm.utils.msa.msa"] = msa_module
    sys.modules["esm.utils.structure"] = structure_module
    sys.modules["esm.utils.structure.molecular_complex"] = molecular_complex
    sys.modules["esm.models"].esmfold2 = esmfold2
    sys.modules["esm.utils.msa"].msa = msa_module
    sys.modules["esm.utils"].msa = sys.modules["esm.utils.msa"]
    sys.modules["esm.utils"].structure = structure_module
    sys.modules["esm"].models = sys.modules["esm.models"]
    sys.modules["esm"].utils = sys.modules["esm.utils"]


if os.environ.get("PREFLIGHT_FAKE_MODEL") == "1":
    _install_torch()
    _install_transformers()
    _install_esm()
'''


FAKE_PROTENIX_SOURCE = '''#!/usr/bin/env python3
# Stands in for the protenix console script. It writes the output tree the arm
# walks, so the arm's own subprocess call, its seed_outputs glob, its
# token_pair_pae read and its file copy all run for real.

import json
import os
import sys
from pathlib import Path

import preflight_fixture

argv = sys.argv[1:]
Path(os.environ["PREFLIGHT_PROTENIX_ARGV"]).write_text(json.dumps(argv, indent=2) + "\\n")


def value_of(flag):
    return argv[argv.index(flag) + 1] if flag in argv else None


document = json.loads(Path(value_of("--input")).read_text())
job = document[0]
name = job["name"]
chains = [
    (entity["proteinChain"]["id"][0], len(entity["proteinChain"]["sequence"]))
    for entity in job["sequences"]
]
size = sum(count for _, count in chains)

failing = os.environ.get("PREFLIGHT_FAIL_JOB_NAME")
if failing and name == failing:
    # The point of the whole exercise. A non-zero exit here must produce a failed
    # row per seed and let the next candidate run.
    print("the preflight stand-in refused this job on purpose", file=sys.stderr)
    raise SystemExit(3)

out_dir = Path(value_of("--out_dir"))
cif = preflight_fixture.complex_cif(chains)
pae = preflight_fixture.pae_rows(size)
for seed in value_of("--seeds").split(","):
    seed_dir = out_dir / name / ("seed_" + seed)
    seed_dir.mkdir(parents=True, exist_ok=True)
    (seed_dir / (name + "_sample_0.cif")).write_text(cif)
    (seed_dir / (name + "_full_data_sample_0.json")).write_text(
        json.dumps({"token_pair_pae": pae})
    )
    (seed_dir / (name + "_summary_confidence_sample_0.json")).write_text(
        json.dumps(
            {
                "plddt": 0.84,
                "ptm": 0.71,
                "iptm": 0.63,
                "ranking_score": 0.67,
                "has_clash": False,
            }
        )
    )
print("preflight protenix stand-in wrote " + name)
'''


# --- reading the things this script must not invent --------------------------


def executor_measurement_fields() -> list[str]:
    """Return MEASUREMENT_SOURCE_FIELDS as the executor spells it.

    Read rather than copied. A field renamed in the executor and not here would
    otherwise make the preflight pass a measurement a paid run rejects.
    """
    source = EXECUTOR_PATH.read_text()
    match = re.search(
        r"^MEASUREMENT_SOURCE_FIELDS = \((.*?)^\)$", source, re.MULTILINE | re.DOTALL
    )
    if match is None:
        raise SystemExit(f"MEASUREMENT_SOURCE_FIELDS was not found in {EXECUTOR_PATH}")
    return re.findall(r'"([a-z0-9_]+)"', match.group(1))


def executor_hotspot_sources() -> list[str]:
    """Return HOTSPOT_SOURCES as the executor spells it.

    Read rather than copied, for the same reason the measurement field list is:
    a source renamed in one place and not the other would let the preflight pass
    a row a paid run rejects.
    """
    source = EXECUTOR_PATH.read_text()
    match = re.search(r"^HOTSPOT_SOURCES = \((.*?)\)$", source, re.MULTILINE | re.DOTALL)
    if match is None:
        raise SystemExit(f"HOTSPOT_SOURCES was not found in {EXECUTOR_PATH}")
    return re.findall(r'"([a-z0-9_]+)"', match.group(1))


def registered_site_metric_basis() -> str:
    """Return one registered site_metric_basis, from the observation schema."""
    schema = json.loads(OBSERVATION_SCHEMA_PATH.read_text())
    values = schema["properties"]["site_metric_basis"]["enum"]
    return str(values[-1])


def profile_adapters(profile_path: Path) -> dict[str, dict[str, Any]]:
    """Return every adapter record, base profile merged with the overlay."""
    profile = json.loads(profile_path.read_text())
    base_name = profile.get("base_profile")
    merged: dict[str, dict[str, Any]] = {}
    if base_name:
        base = json.loads((profile_path.parent / base_name).read_text())
        for adapter in base.get("adapters", []):
            merged[str(adapter["adapter_id"])] = dict(adapter)
    overlay = profile.get("overlay", {})
    for adapter in overlay.get("adapter_overrides", []):
        adapter_id = str(adapter["adapter_id"])
        merged.setdefault(adapter_id, {})
        merged[adapter_id].update(adapter)
    for adapter in profile.get("adapters", []):
        adapter_id = str(adapter["adapter_id"])
        merged.setdefault(adapter_id, {})
        merged[adapter_id].update(adapter)
    return merged


def render_argv(template: list[str], context: dict[str, Any]) -> tuple[list[str], list[str]]:
    """Use the executor renderer and report values the preflight cannot run."""
    unfilled: list[str] = []
    for item in template:
        if item == UNRESOLVED:
            unfilled.append("the whole argv element is " + UNRESOLVED)
        unfilled.extend(
            token for token in EXECUTOR.TOKEN_RE.findall(item) if token not in context
        )
    if unfilled:
        return list(template), unfilled
    return EXECUTOR.render_argv(template, context), []


# --- the fixture -------------------------------------------------------------


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    return path


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))
    return path


def fill(value: Any, replacement: Any) -> Any:
    """Return the template's value, or the local replacement where it is unresolved.

    The template spells an unresolved list as a one-element list holding the
    placeholder, so a bare equality check would carry `__REQUIRED__` through into
    a site definition and fail forty lines later inside the metric.
    """
    if value == UNRESOLVED:
        return replacement
    if isinstance(value, list) and any(item == UNRESOLVED for item in value):
        return replacement
    return value


class Fixture:
    """Every file the adapters read, written under one working directory."""

    def __init__(self, work_dir: Path) -> None:
        self.root = work_dir
        self.fakes_dir = work_dir / "fakes"
        self.inputs_dir = work_dir / "inputs"
        self.artifact_root = work_dir / "artifacts"
        self.receipts_dir = self.artifact_root / "receipts"
        self.measurement_fields = executor_measurement_fields()
        self.hotspot_sources = executor_hotspot_sources()
        self.candidates: list[dict[str, Any]] = []
        self.target_id = ""
        self.target_chain = ""
        self.binder_chain = ""

    # -- the stand-in files

    def write_fakes(self) -> None:
        self.fakes_dir.mkdir(parents=True, exist_ok=True)
        (self.fakes_dir / "preflight_fixture.py").write_text(FIXTURE_MODULE_SOURCE.lstrip())
        (self.fakes_dir / "sitecustomize.py").write_text(SITECUSTOMIZE_SOURCE.lstrip())
        bin_dir = self.fakes_dir / "bin"
        bin_dir.mkdir(parents=True, exist_ok=True)
        protenix = bin_dir / "protenix"
        protenix.write_text(FAKE_PROTENIX_SOURCE)
        protenix.chmod(0o755)

    # -- the campaign inputs

    def sequence(self, length: int, letter: str) -> str:
        return letter * length

    def build(self, adapters: dict[str, dict[str, Any]]) -> None:
        self.write_fakes()
        template = json.loads(CAMPAIGN_TEMPLATE_PATH.read_text())
        target_template = template["targets"][0]
        binder_template = template["binder"]
        self.target_chain = str(binder_template["target_chain_id"])
        self.binder_chain = str(binder_template["binder_chain_id"])
        self.target_id = str(target_template["target_id"])

        sys.path.insert(0, str(self.fakes_dir))
        import preflight_fixture  # noqa: PLC0415

        self.inputs_dir.mkdir(parents=True, exist_ok=True)
        target_sequence = self.sequence(TARGET_LENGTH, "A")
        target_fasta = self.inputs_dir / "target.fasta"
        target_fasta.write_text(f">{self.target_id}\n{target_sequence}\n")

        # A two-record a3m, so clean_a3m has a second row to keep and MSA.from_a3m
        # is reached with a real path.
        target_a3m = self.inputs_dir / "target.a3m"
        target_a3m.write_text(
            f">{self.target_id}\n{target_sequence}\n>homolog\n{target_sequence}\n"
        )

        # The target structure is hashed onto every row and is never parsed by an
        # adapter, so the fixture writes the same complex generator's target chain.
        target_structure = self.inputs_dir / "target.cif"
        target_structure.write_text(
            preflight_fixture.complex_cif([(self.target_chain, TARGET_LENGTH)])
        )

        residue_map = write_json(
            self.inputs_dir / "residue-map.json",
            {"source_to_cleaned": {}},
        )

        site_template = dict(target_template["site"])
        site = {
            **site_template,
            "design_residues": fill(
                site_template.get("design_residues"), [f"{self.target_chain}:1-3"]
            ),
            "reference_contact_residues": fill(
                site_template.get("reference_contact_residues"),
                [f"{self.target_chain}:1-2"],
            ),
            # The declared hotspot source. `explicit` is the one the argv can
            # serve today, and the list beside it is the same one
            # `supplemental_arguments` passes on --hotspot-residues, so the
            # config and the argv agree. The template carries neither key, so an
            # absent one falls back the way an unresolved one does.
            "hotspot_source": fill(
                site_template.get("hotspot_source") or UNRESOLVED, self.hotspot_sources[0]
            ),
            "hotspot_residues": fill(
                site_template.get("hotspot_residues") or UNRESOLVED,
                [f"{self.target_chain}:1"],
            ),
            "residue_map_path": str(residue_map),
            "runtime_residue_map_path": str(residue_map),
            "residue_map_sha256": sha256_file(residue_map),
        }
        target = {
            **target_template,
            "site": site,
            "chains": [{"chain_id": self.target_chain, "role": "design-target"}],
            "structure_path": str(target_structure),
            "runtime_structure_path": str(target_structure),
            "structure_sha256": sha256_file(target_structure),
            "source_id": fill(target_template.get("source_id"), "preflight-fixture"),
        }

        # Candidates. Each one gets its own binder sequence and its own design
        # pose, and the pose is the geometry the stand-in predicts, so compute_dockq
        # has a reference it can map.
        candidates_dir = self.inputs_dir / "candidates"
        for index, length in enumerate(CANDIDATE_BINDER_LENGTHS):
            candidate_id = f"preflight-candidate-{index:02d}"
            candidate_dir = candidates_dir / candidate_id
            candidate_dir.mkdir(parents=True, exist_ok=True)
            binder_sequence = self.sequence(length, "G")
            binder_fasta = candidate_dir / "binder.fasta"
            binder_fasta.write_text(f">{candidate_id}\n{binder_sequence}\n")
            design_pose = candidate_dir / "design-pose.cif"
            design_pose.write_text(
                preflight_fixture.complex_cif(
                    [(self.target_chain, TARGET_LENGTH), (self.binder_chain, length)]
                )
            )
            self.candidates.append(
                {
                    "candidate_id": candidate_id,
                    "sequence_path": str(binder_fasta),
                    "sequence_sha256": sha256_file(binder_fasta),
                    "design_pose_path": str(design_pose),
                    "design_pose_sha256": sha256_file(design_pose),
                    "origin_generator": "rfdiffusion",
                    "binder_sequence": binder_sequence,
                }
            )

        manifest_rows = [
            {key: value for key, value in candidate.items() if key != "binder_sequence"}
            for candidate in self.candidates
        ]
        write_jsonl(self.artifact_root / "filters" / "passing-candidates.jsonl", manifest_rows)
        write_jsonl(
            self.artifact_root / "optimization" / "rescore-candidates.jsonl", manifest_rows
        )
        self.receipts_dir.mkdir(parents=True, exist_ok=True)

        self.target_fasta = target_fasta
        self.target_a3m = target_a3m
        self.config_path = write_json(
            self.root / "config.resolved.json",
            self.build_config(template, target, adapters),
        )
        self.plan_path = write_json(self.root / "plan.json", {"schema_version": 1})

    def build_config(
        self,
        template: dict[str, Any],
        target: dict[str, Any],
        adapters: dict[str, dict[str, Any]],
    ) -> dict[str, Any]:
        """Assemble the resolved campaign configuration the adapters read.

        Only the blocks an adapter actually reads are present. Every field name
        here is one an adapter or the executor reads by that name.
        """
        predictors = [
            {
                "id": "esmfold2",
                "adapter_id": "esmfold2-predictor",
                "enabled": True,
                "screen_stage": "cofold-screen-esmfold2",
                "rescore_stage": "cofold-rescore-esmfold2",
            },
            {
                "id": "esmfold2-fast",
                "adapter_id": "esmfold2-fast-predictor",
                "enabled": True,
                "screen_stage": "cofold-screen-esmfold2-fast",
                "rescore_stage": "cofold-rescore-esmfold2-fast",
            },
            {
                "id": "protenix-v2",
                "adapter_id": "protenix-v2-predictor",
                "enabled": True,
                "screen_stage": "cofold-screen-protenix-v2",
                "rescore_stage": "cofold-rescore-protenix-v2",
            },
        ]
        stages: list[dict[str, Any]] = []
        for predictor in predictors:
            stages.append(
                {
                    "stage_id": predictor["screen_stage"],
                    "adapter_id": predictor["adapter_id"],
                    "mode": "smoke_scale",
                    "outputs": [
                        {
                            "artifact_id": f"{predictor['id']}-screen",
                            "artifact_type": "raw-prediction-manifest",
                            "kind": "jsonl",
                            "path_template": "{{attempt_dir}}/{{phase}}/cofold-observations.jsonl",
                        }
                    ],
                }
            )
        stages.append(
            {
                "stage_id": "score-screen",
                "adapter_id": "interface-scorer",
                "mode": "single",
                "outputs": [
                    {
                        "artifact_id": "screen-score-table",
                        "artifact_type": "screen-score-table",
                        "kind": "jsonl",
                        "path_template": "{{attempt_dir}}/{{phase}}/screen-score-table.jsonl",
                    }
                ],
            }
        )
        return {
            "schema_version": template["schema_version"],
            "campaign_id": "preflight-adapters",
            "run_id": "preflight",
            "binder": {
                "target_chain_id": self.target_chain,
                "binder_chain_id": self.binder_chain,
            },
            "targets": [target],
            "controls": {"positive": [], "negative": []},
            "cofold": {
                "predictors": predictors,
                "screen_seeds": [0],
                "rescore_seeds": [0, 1, 2, 3, 4],
            },
            "adapters": [
                {
                    "adapter_id": adapter_id,
                    "model_revision": adapters.get(adapter_id, {}).get(
                        "model_revision", UNRESOLVED
                    ),
                }
                for adapter_id in sorted(adapters)
            ],
            "scoring": {"implementations": {"site_metric_basis": registered_site_metric_basis()}},
            "stages": stages,
        }

    # -- the environment every adapter runs in

    def environment(self, *, fake_model: bool, extra: dict[str, str] | None = None) -> dict[str, str]:
        env = dict(os.environ)
        # The target environment exports this, and it is what turns off the
        # implicit script-directory entry the adapters restore by hand.
        env["PYTHONSAFEPATH"] = "1"
        env["PYTHONPATH"] = os.pathsep.join((str(self.fakes_dir), str(PACKAGE_ROOT.parent)))
        env["PATH"] = str(self.fakes_dir / "bin") + os.pathsep + env.get("PATH", "")
        env["PREFLIGHT_FAKE_MODEL"] = "1" if fake_model else "0"
        env.update(extra or {})
        return env


# --- running one adapter -----------------------------------------------------


class Check:
    """One named assertion and its outcome."""

    def __init__(self, name: str, ok: bool, detail: str = "") -> None:
        self.name = name
        self.ok = ok
        self.detail = detail


class AdapterResult:
    def __init__(self, adapter_id: str) -> None:
        self.adapter_id = adapter_id
        self.checks: list[Check] = []
        self.notes: list[str] = []

    def check(self, name: str, ok: bool, detail: str = "") -> bool:
        self.checks.append(Check(name, bool(ok), detail))
        return bool(ok)

    def note(self, text: str) -> None:
        self.notes.append(text)

    @property
    def ok(self) -> bool:
        return all(check.ok for check in self.checks)


def run_argv(argv: list[str], env: dict[str, str], log_path: Path) -> subprocess.CompletedProcess:
    completed = subprocess.run(
        argv,
        cwd=str(REPO_ROOT),
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(
        "argv: " + json.dumps(argv, indent=2) + "\n"
        f"returncode: {completed.returncode}\n"
        "--- stdout ---\n" + completed.stdout + "\n--- stderr ---\n" + completed.stderr
    )
    return completed


def attempt_context(fixture: Fixture, stage_id: str, phase: str, count: int) -> dict[str, Any]:
    """Return the token values the executor supplies, with the same shapes.

    attempt_dir follows artifact_root/stages/{stage_id}/attempts/{attempt_id},
    which is where run_stage puts it.
    """
    config = json.loads(fixture.config_path.read_text())
    target = EXECUTOR.primary_target(config)
    attempt_id = uuid.uuid4().hex
    attempt_dir = fixture.artifact_root / "stages" / stage_id / "attempts" / attempt_id
    return {
        "target_id": str(target["target_id"]),
        "stage_id": stage_id,
        "phase": phase,
        "count": count,
        "attempt_id": attempt_id,
        "attempt_dir": str(attempt_dir),
        "receipts_dir": str(fixture.receipts_dir),
        "artifact_root": str(fixture.artifact_root),
        "run_root": str(fixture.root),
        "config_path": str(fixture.config_path),
        "plan_path": str(fixture.plan_path),
        "optimization_round": 0,
        "python_executable": sys.executable,
    }


# --- the contract assertions -------------------------------------------------


ARTIFACT_ROW_FIELDS = (
    "predicted_complex_path",
    "predicted_complex_sha256",
    "pae_path",
    "pae_sha256",
    "metric_source_path",
    "metric_source_sha256",
)


def slug_for(target_id: str, candidate_id: str, predictor: str, phase: str, seed: Any) -> str:
    """Rebuild the artifact directory name from the five identifying fields."""
    unsafe = re.compile(r"[^a-zA-Z0-9_.-]")
    return "-".join(
        unsafe.sub("_", str(value))
        for value in (target_id, candidate_id, predictor, phase, seed)
    )


def assert_contract(
    result: AdapterResult,
    *,
    fixture: Fixture,
    rows: list[dict[str, Any]],
    attempt_dir: Path,
    run_phase: str,
    expected_row_count: int,
) -> None:
    """Check every row against what the executor requires of it."""
    result.check(
        "manifest holds one row per planned prediction",
        len(rows) == expected_row_count,
        f"{len(rows)} rows against {expected_row_count} planned",
    )
    scored = [row for row in rows if row.get("status") == "scored"]
    failed = [row for row in rows if row.get("status") == "failed"]
    result.check(
        "every row carries a status the schema registers",
        len(scored) + len(failed) == len(rows),
        f"{len(rows) - len(scored) - len(failed)} rows carry another status",
    )
    result.check("at least one row was scored", bool(scored), f"{len(scored)} scored")

    fields = fixture.measurement_fields
    for row in scored:
        label = f"{row.get('candidate_id')} seed {row.get('seed')}"
        missing = [name for name in ARTIFACT_ROW_FIELDS if not row.get(name)]
        if not result.check(
            f"scored row {label} carries the six path and hash fields",
            not missing,
            "missing " + ", ".join(missing) if missing else "",
        ):
            continue

        slug = slug_for(
            row["target_id"], row["candidate_id"], row["predictor"], row["phase"], row["seed"]
        )
        expected_dir = attempt_dir / run_phase / "prediction-artifacts" / slug
        complex_path = expected_dir / "complex.cif"
        pae_path = expected_dir / "pae.json"
        metric_path = expected_dir / "measurement-source.json"
        present = [path.is_file() for path in (complex_path, pae_path, metric_path)]
        if not result.check(
            f"scored row {label} wrote the three files at the contract path",
            all(present),
            f"under {expected_dir}",
        ):
            continue
        result.check(
            f"scored row {label} points at the files it wrote",
            Path(row["predicted_complex_path"]).resolve() == complex_path.resolve()
            and Path(row["pae_path"]).resolve() == pae_path.resolve()
            and Path(row["metric_source_path"]).resolve() == metric_path.resolve(),
        )
        result.check(
            f"scored row {label} hashes match the files on disk",
            row["predicted_complex_sha256"] == sha256_file(complex_path)
            and row["pae_sha256"] == sha256_file(pae_path)
            and row["metric_source_sha256"] == sha256_file(metric_path),
        )

        pae_document = json.loads(pae_path.read_text())
        matrices = [
            key
            for key, value in pae_document.items()
            if isinstance(value, list) and value and isinstance(value[0], list)
        ]
        result.check(
            f"scored row {label} pae.json carries the matrix under pae and nothing else",
            matrices == ["pae"],
            "matrix keys " + ", ".join(sorted(matrices)),
        )

        measurement = json.loads(metric_path.read_text()).get("measurement", {})
        absent = [name for name in fields if name not in measurement]
        expected_derived = {
            "predicted_target_chain_id": fixture.target_chain,
            "predicted_binder_chain_id": fixture.binder_chain,
            "reference_target_chain_id": fixture.target_chain,
            "reference_binder_chain_id": fixture.binder_chain,
        }
        derived_mismatch = [
            name for name, expected in expected_derived.items() if measurement.get(name) != expected
        ]
        result.check(
            f"scored row {label} measurement carries all {len(fields)} executor fields",
            not absent and not derived_mismatch,
            (
                "missing " + ", ".join(absent)
                if absent
                else "derived chain mismatch at " + ", ".join(derived_mismatch)
                if derived_mismatch
                else ""
            ),
        )
        null = [name for name in fields if measurement.get(name) is None]
        result.check(
            f"scored row {label} measurement carries no null in a required field",
            not null,
            "null at " + ", ".join(null) if null else "",
        )
        result.check(
            f"scored row {label} measurement binds to the complex the row names",
            measurement.get("predicted_complex_sha256") == row["predicted_complex_sha256"],
        )
        # Which of the three sources produced hotspot_recovery. A row that does
        # not say cannot be audited, and a name nobody registered means the
        # metric and the executor disagree about what the sources are.
        result.check(
            f"scored row {label} measurement names a registered hotspot source",
            measurement.get("site_hotspot_source") in fixture.hotspot_sources,
            f"site_hotspot_source is {measurement.get('site_hotspot_source')!r}",
        )

    for row in failed:
        label = f"{row.get('candidate_id')} seed {row.get('seed')}"
        carried = [name for name in ARTIFACT_ROW_FIELDS if name in row]
        result.check(
            f"failed row {label} carries no artifact path",
            not carried,
            "carries " + ", ".join(carried) if carried else "",
        )
        result.check(
            f"failed row {label} names its failure",
            bool(row.get("failure_code")) and bool(row.get("failure_reason")),
            str(row.get("failure_code")),
        )


def assert_failure_path(
    result: AdapterResult,
    *,
    rows: list[dict[str, Any]],
    failing_candidate_id: str,
    later_candidate_ids: list[str],
) -> None:
    """Check that one bad candidate cost one candidate and not the job."""
    failing = [row for row in rows if row.get("candidate_id") == failing_candidate_id]
    result.check(
        "the refused candidate produced a row per planned prediction",
        bool(failing),
        f"{len(failing)} rows",
    )
    result.check(
        "the refused candidate is marked failed rather than dropped",
        bool(failing) and all(row.get("status") == "failed" for row in failing),
    )
    later = [row for row in rows if row.get("candidate_id") in later_candidate_ids]
    result.check(
        "the candidates after the refused one still ran",
        bool(later) and all(row.get("status") == "scored" for row in later),
        f"{sum(1 for row in later if row.get('status') == 'scored')} of {len(later)} scored",
    )


# --- published predictor modes ----------------------------------------------


def supplemental_arguments(fixture: Fixture, adapter_id: str) -> list[str]:
    """Return the arguments the arm declares and the argv template has no token for.

    The campaign configuration carries no target sequence, no hotspot list and no
    a3m path, so each arm exposes them as arguments. `command_argv_template`
    passes none of them.
    """
    target = fixture.target_id
    extra = [
        "--target-sequence",
        f"{target}={fixture.target_fasta}",
        "--hotspot-residues",
        f"{target}={fixture.target_chain}:1",
    ]
    if adapter_id == "esmfold2-predictor":
        extra += ["--target-msa-a3m", f"{target}={fixture.target_a3m}"]
    if adapter_id == "protenix-v2-predictor":
        extra += ["--target-unpaired-msa-a3m", f"{target}={fixture.target_a3m}"]
    return extra


def run_arm(
    fixture: Fixture,
    adapters: dict[str, dict[str, Any]],
    adapter_id: str,
    predictor_id: str,
    stage_id: str,
) -> tuple[AdapterResult, Path | None]:
    """Run one arm twice and check the manifest it wrote the second time."""
    result = AdapterResult(adapter_id)
    adapter = adapters.get(adapter_id, {})
    command_template = adapter.get("command_argv_template")
    parser_template = adapter.get("parser_argv_template")
    if not command_template:
        result.check("the profile carries a command_argv_template", False, "absent")
        return result, None

    candidates = fixture.candidates
    failing = candidates[FAILING_CANDIDATE_INDEX]
    later_ids = [item["candidate_id"] for item in candidates[FAILING_CANDIDATE_INDEX + 1 :]]
    expected_rows = len(candidates)  # one target, one screen seed

    # Variant one: the argv the dispatcher renders today, and nothing else.
    bare_context = attempt_context(fixture, stage_id, RUN_PHASE, len(candidates))
    bare_argv, unfilled = render_argv(command_template, bare_context)
    result.check(
        "every token in command_argv_template has a value",
        not unfilled,
        "unfilled: " + ", ".join(sorted(set(unfilled))) if unfilled else "",
    )
    if unfilled:
        return result, None
    bare = run_argv(bare_argv, fixture.environment(fake_model=True), fixture.root / "logs" / f"{adapter_id}-bare.log")
    bare_manifest = Path(bare_context["attempt_dir"]) / RUN_PHASE / "cofold-observations.jsonl"
    bare_rows = (
        [json.loads(line) for line in bare_manifest.read_text().splitlines() if line.strip()]
        if bare_manifest.is_file()
        else []
    )
    bare_codes = sorted({str(row.get("failure_code")) for row in bare_rows if row.get("status") == "failed"})
    result.note(
        f"template argv alone: exit {bare.returncode}, {len(bare_rows)} rows, "
        f"{sum(1 for row in bare_rows if row.get('status') == 'scored')} scored"
        + (f", failure codes {', '.join(bare_codes)}" if bare_codes else "")
    )

    # Variant two: the same argv with the per-target arguments the template has
    # no token for. This is the only variant that reaches the fold and the write.
    context = attempt_context(fixture, stage_id, RUN_PHASE, len(candidates))
    argv, _ = render_argv(command_template, context)
    argv += supplemental_arguments(fixture, adapter_id)
    fake_log = fixture.root / "logs" / f"{adapter_id}-fake-calls.jsonl"
    fake_log.parent.mkdir(parents=True, exist_ok=True)
    env = fixture.environment(
        fake_model=True,
        extra={
            "PREFLIGHT_FAKE_LOG": str(fake_log),
            "PREFLIGHT_FAIL_BINDER_SEQUENCE": failing["binder_sequence"],
            "PREFLIGHT_FAIL_JOB_NAME": f"{fixture.target_id}-{failing['candidate_id']}",
            "PREFLIGHT_PROTENIX_ARGV": str(fixture.root / "logs" / "protenix-argv.json"),
        },
    )
    completed = run_argv(argv, env, fixture.root / "logs" / f"{adapter_id}-run.log")
    result.check(
        "the arm exits zero when it produced a row per prediction",
        completed.returncode == 0,
        f"exit {completed.returncode}; see logs/{adapter_id}-run.log",
    )

    attempt_dir = Path(context["attempt_dir"])
    manifest = attempt_dir / RUN_PHASE / "cofold-observations.jsonl"
    if not result.check(
        "the arm wrote its manifest at the path the stage contract names",
        manifest.is_file(),
        str(manifest),
    ):
        return result, None
    rows = [json.loads(line) for line in manifest.read_text().splitlines() if line.strip()]
    assert_contract(
        result,
        fixture=fixture,
        rows=rows,
        attempt_dir=attempt_dir,
        run_phase=RUN_PHASE,
        expected_row_count=expected_rows,
    )
    assert_failure_path(
        result,
        rows=rows,
        failing_candidate_id=failing["candidate_id"],
        later_candidate_ids=later_ids,
    )
    result.check(
        "every scored row names the predictor the campaign registered",
        all(row.get("predictor") == predictor_id for row in rows),
    )

    # What the stand-in saw. A wrong chain order inverts ipSAE without raising,
    # so the order the arm folded in is asserted rather than assumed.
    if adapter_id.startswith("esmfold2"):
        calls = [
            json.loads(line)
            for line in fake_log.read_text().splitlines()
            if line.strip()
        ] if fake_log.is_file() else []
        folds = [call for call in calls if call.get("call") == "fold"]
        result.check(
            "the arm folded the target chain first and the binder second",
            bool(folds)
            and all(
                [chain for chain, _ in call["chains"]]
                == [fixture.target_chain, fixture.binder_chain]
                for call in folds
            ),
        )
        expects_msa = adapter_id == "esmfold2-predictor"
        result.check(
            "the arm gave the target an MSA only where its checkpoint has an encoder",
            bool(folds)
            and all(call["msa_present"] == [expects_msa, False] for call in folds),
            "MSA policy for " + adapter_id,
        )
        loads = [call for call in calls if call.get("call") == "from_pretrained"]
        expected_checkpoint = (
            "biohub/ESMFold2-Fast" if adapter_id == "esmfold2-fast-predictor" else "biohub/ESMFold2"
        )
        result.check(
            "the arm loaded the checkpoint its ArmSpec names",
            bool(loads) and all(call["checkpoint"] == expected_checkpoint for call in loads),
            ", ".join(sorted({call["checkpoint"] for call in loads})),
        )

    if adapter_id == "protenix-v2-predictor":
        argv_path = fixture.root / "logs" / "protenix-argv.json"
        if result.check("the arm called the protenix console script", argv_path.is_file()):
            called = json.loads(argv_path.read_text())
            result.check(
                "the Protenix call carries input, output, and seed arguments",
                all(
                    flag in called
                    and called.index(flag) + 1 < len(called)
                    and bool(called[called.index(flag) + 1])
                    for flag in ("--input", "--out_dir", "--seeds")
                ),
                " ".join(called),
            )
            result.check(
                "the call carries the flags the published row settles",
                "--step" not in called
                and "--dtype" not in called
                and called[called.index("--sample") + 1] == "1"
                and called[called.index("--cycle") + 1] == "10"
                and called[called.index("--need_atom_confidence") + 1] == "true"
                and called[called.index("--use_msa") + 1] == "true",
                " ".join(called),
            )

    # The dispatcher runs the parser right after the command, against the same
    # tokens. A parser that cannot find what the command wrote fails the stage.
    if parser_template:
        parse_argv, parse_unfilled = render_argv(parser_template, context)
        if result.check(
            "every token in parser_argv_template has a value",
            not parse_unfilled,
            ", ".join(sorted(set(parse_unfilled))),
        ):
            parsed = run_argv(
                parse_argv, fixture.environment(fake_model=False), fixture.root / "logs" / f"{adapter_id}-parse.log"
            )
            parser_result_path = attempt_dir / RUN_PHASE / "parser-result.json"
            result.check(
                "the parser reports ok on the outputs the command wrote",
                parsed.returncode == 0
                and parser_result_path.is_file()
                and json.loads(parser_result_path.read_text()).get("ok") is True,
                f"exit {parsed.returncode}",
            )

    return result, manifest


def run_fast_arm_msa_guard(fixture: Fixture, adapters: dict[str, dict[str, Any]]) -> AdapterResult:
    """Check that the Fast arm refuses an a3m rather than ignoring one.

    The released Fast checkpoint has no MSA encoder, so an accepted a3m would be a
    silent no-op and the numbers would be single-sequence numbers under an MSA
    label. The arm's own docstring says argparse rejects the argument.
    """
    result = AdapterResult("esmfold2-fast-predictor a3m guard")
    adapter = adapters.get("esmfold2-fast-predictor", {})
    template = adapter.get("command_argv_template")
    if not template:
        result.check("the profile carries a command_argv_template", False, "absent")
        return result
    context = attempt_context(fixture, "cofold-screen-esmfold2-fast", RUN_PHASE, 1)
    argv, unfilled = render_argv(template, context)
    if unfilled:
        result.check(
            "every token in the Fast arm template resolves",
            False,
            "unresolved: " + ", ".join(unfilled),
        )
        return result
    argv += ["--target-msa-a3m", f"{fixture.target_id}={fixture.target_a3m}"]
    completed = run_argv(
        argv, fixture.environment(fake_model=True), fixture.root / "logs" / "fast-a3m-guard.log"
    )
    result.check(
        "the Fast arm rejects --target-msa-a3m instead of ignoring it",
        completed.returncode != 0 and "--target-msa-a3m" in completed.stderr,
        f"exit {completed.returncode}",
    )
    return result


# --- the reducer -------------------------------------------------------------


def run_scorer(
    fixture: Fixture,
    adapters: dict[str, dict[str, Any]],
    manifests: dict[str, Path],
) -> AdapterResult:
    """Run interface_scorer over the published-mode manifests."""
    result = AdapterResult("interface-scorer")
    adapter = adapters.get("interface-scorer", {})
    template = adapter.get("command_argv_template") or []
    context = attempt_context(fixture, "score-screen", SCORER_RUN_PHASE, 1)
    _, unfilled = render_argv(template, context)
    if unfilled or not template:
        result.note(
            "the profile does not define this adapter's argv, so the command the "
            "dispatcher would run is unknown: command_argv_template is "
            + json.dumps(template)
        )
        # The argument surface the script's own parser declares is the same eight
        # options every other adapter takes, so the reducer is still exercised.
        argv = [
            sys.executable,
            "-m",
            "claude_binder.adapters.interface_scorer",
            "run",
            "--stage",
            context["stage_id"],
            "--phase",
            context["phase"],
            "--count",
            str(context["count"]),
            "--attempt-dir",
            context["attempt_dir"],
            "--receipts-dir",
            context["receipts_dir"],
            "--artifact-root",
            context["artifact_root"],
            "--config",
            context["config_path"],
            "--plan",
            context["plan_path"],
        ]
    else:
        argv, _ = render_argv(template, context)

    # The reducer reads each arm's stage receipt. The shape copies what run_stage
    # writes: a combined output manifest whose artifacts carry an artifact_type,
    # the phase they were written under, and one file record per manifest.
    missing = [stage_id for stage_id, path in manifests.items() if not path or not path.is_file()]
    if not result.check(
        "each arm left a manifest for the reducer to read",
        not missing,
        "no manifest from " + ", ".join(missing) if missing else "",
    ):
        return result
    for stage_id, path in manifests.items():
        write_json(
            fixture.receipts_dir / f"{stage_id}.json",
            {
                "schema_version": 1,
                "stage_id": stage_id,
                "output_manifest": {
                    "schema_version": 1,
                    "stage_id": stage_id,
                    "artifacts": [
                        {
                            "artifact_id": f"{stage_id}-manifest",
                            "artifact_type": "raw-prediction-manifest",
                            "kind": "jsonl",
                            "phase": RUN_PHASE,
                            "files": [{"path": str(path), "sha256": sha256_file(path)}],
                        }
                    ],
                },
            },
        )

    completed = run_argv(
        argv, fixture.environment(fake_model=False), fixture.root / "logs" / "interface-scorer-run.log"
    )
    result.check(
        "the reducer exits zero",
        completed.returncode == 0,
        f"exit {completed.returncode}; see logs/interface-scorer-run.log",
    )
    table = Path(context["attempt_dir"]) / SCORER_RUN_PHASE / "screen-score-table.jsonl"
    if not result.check("the reducer wrote the score table", table.is_file(), str(table)):
        return result
    rows = [json.loads(line) for line in table.read_text().splitlines() if line.strip()]
    expected = sum(
        len([line for line in path.read_text().splitlines() if line.strip()])
        for path in manifests.values()
    )
    result.check(
        "the score table holds one row per prediction the arms wrote",
        len(rows) == expected,
        f"{len(rows)} rows against {expected}",
    )
    scored = [row for row in rows if row.get("status") == "scored"]
    fields = fixture.measurement_fields
    for row in scored:
        absent = [name for name in fields if name not in row]
        if absent:
            result.check(
                f"score row {row.get('candidate_id')} {row.get('predictor')} "
                f"carries all {len(fields)} measurement fields",
                False,
                "missing " + ", ".join(absent),
            )
            break
    else:
        result.check(
            f"every scored row carries all {len(fields)} measurement fields",
            bool(scored),
            f"{len(scored)} scored rows",
        )
    unnamed = [
        row for row in scored if row.get("site_hotspot_source") not in fixture.hotspot_sources
    ]
    result.check(
        "every scored row names a registered hotspot source",
        bool(scored) and not unnamed,
        f"{len(unnamed)} of {len(scored)} rows do not",
    )
    result.check(
        "a failed prediction still produced a row",
        any(row.get("status") == "failed" for row in rows),
    )

    parser_template = adapter.get("parser_argv_template") or []
    _, parse_unfilled = render_argv(parser_template, context)
    if parser_template and not parse_unfilled:
        parse_argv, _ = render_argv(parser_template, context)
        parsed = run_argv(
            parse_argv,
            fixture.environment(fake_model=False),
            fixture.root / "logs" / "interface-scorer-parse.log",
        )
        result.check("the reducer's parser exits zero", parsed.returncode == 0)
    else:
        result.note(
            "the profile does not define this adapter's parser argv either, so the "
            "parse step was not run as the dispatcher would run it"
        )
    return result


# --- reporting ---------------------------------------------------------------


def report(results: list[AdapterResult]) -> int:
    print()
    print("=" * 78)
    print("PREFLIGHT SUMMARY")
    print("=" * 78)
    failed_total = 0
    for result in results:
        passed = sum(1 for check in result.checks if check.ok)
        failures = [check for check in result.checks if not check.ok]
        failed_total += len(failures)
        state = "PASS" if result.ok else "FAIL"
        print(f"\n[{state}] {result.adapter_id}: {passed} of {len(result.checks)} checks passed")
        for note in result.notes:
            print(f"    note  {note}")
        for check in failures:
            detail = f" ({check.detail})" if check.detail else ""
            print(f"    FAIL  {check.name}{detail}")
    print()
    if failed_total:
        print(f"{failed_total} checks failed. No adapter is ready for a paid run.")
    else:
        print("Every check passed.")
    return 1 if failed_total else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--profile",
        type=Path,
        default=DEFAULT_PROFILE_PATH,
        help="The profile whose command_argv_template every command is built from.",
    )
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=None,
        help="Where the fixture and every log go. A fresh temporary directory by default.",
    )
    args = parser.parse_args()

    work_dir = args.work_dir or Path(tempfile.mkdtemp(prefix="binder-preflight-"))
    work_dir.mkdir(parents=True, exist_ok=True)
    if shutil.which("python3") is None:
        raise SystemExit("the rendered argv starts with python3 and PATH has none")

    adapters = profile_adapters(args.profile)
    fixture = Fixture(work_dir)
    fixture.build(adapters)
    print(f"profile:   {args.profile}")
    print(f"work dir:  {work_dir}")
    print(f"repo root: {REPO_ROOT}")
    print(f"executor field list: {len(fixture.measurement_fields)} names from {EXECUTOR_PATH.name}")
    print()

    arms = [
        ("esmfold2-predictor", "esmfold2", "cofold-screen-esmfold2"),
        ("esmfold2-fast-predictor", "esmfold2-fast", "cofold-screen-esmfold2-fast"),
        ("protenix-v2-predictor", "protenix-v2", "cofold-screen-protenix-v2"),
    ]
    results: list[AdapterResult] = []
    manifests: dict[str, Path] = {}
    for adapter_id, predictor_id, stage_id in arms:
        print(f"running {adapter_id} ...")
        result, manifest = run_arm(fixture, adapters, adapter_id, predictor_id, stage_id)
        results.append(result)
        if manifest is not None:
            manifests[stage_id] = manifest

    print("running the Fast arm's a3m guard ...")
    results.append(run_fast_arm_msa_guard(fixture, adapters))

    print("running interface-scorer ...")
    results.append(run_scorer(fixture, adapters, manifests))

    return report(results)


if __name__ == "__main__":
    raise SystemExit(main())
