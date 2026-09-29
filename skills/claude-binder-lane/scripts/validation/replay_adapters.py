#!/usr/bin/env python3
"""Replay released co-fold predictions through the real ranking adapters.

The harness replaces only the model forward pass. ESMFold2 receives a fake
model object whose fold result comes from a released model.cif and pae.npz.
Protenix receives a fake ``protenix pred`` executable that writes the same
released files into the output tree the real adapter reads. Parsing, chain
mapping, metric calculation, artifact writing, and parser commands remain the
repository implementations.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import textwrap
import time
from collections import Counter
from pathlib import Path
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import fetch_remote_zip_member as remote_zip

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
    raise RuntimeError(
        "claude_binder package not found. Reinstall this skill; it does not "
        "ship the package these scripts import.")


PACKAGE_ROOT = resolve_package_root()
if str(PACKAGE_ROOT.parent) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT.parent))

from claude_binder.adapters import binder_metrics

RELEASE_URL = (
    "https://huggingface.co/datasets/Anthropic/claude-protein-binder-design/"
    "resolve/main/structure_and_pae/"
    "protein_binder_design_structure_and_pae_release.zip"
)
ARCHIVE_SIZE = 74_481_504_534
CAMPAIGN_TEMPLATE_PATH = PACKAGE_ROOT / "data" / "templates" / "campaign.template.json"
OBSERVATION_SCHEMA_PATH = PACKAGE_ROOT / "data" / "schemas" / "cofold-observation.schema.json"
DEFAULT_PROFILE_PATH = PACKAGE_ROOT / "data" / "templates" / "profiles" / "modal-draft.json"
# Both of these are read from the cache directory when a reader has already put
# a copy there, and neither is required. An absent predictions.parquet is
# range-fetched from the release, and an absent central directory is read over
# HTTP, so a first run needs no local file at all.
PREDICTIONS_CACHE_NAME = "predictions.parquet"
CENTRAL_DIRECTORY_CACHE_NAME = "cdir.bin"

UNRESOLVED = "__REQUIRED__"
TOKEN_RE = re.compile(r"\{\{([a-z_]+)\}\}")
REQUIRED_ADAPTERS = (
    {
        "predictor_id": "esmfold2-fast",
        "adapter_id": "esmfold2-fast-predictor",
        "release_arm": "ef2fast",
        "stage_id": "cofold-rescore-esmfold2-fast",
    },
    {
        "predictor_id": "esmfold2",
        "adapter_id": "esmfold2-predictor",
        "release_arm": "ef2full",
        "stage_id": "cofold-rescore-esmfold2",
    },
    {
        "predictor_id": "protenix-v2",
        "adapter_id": "protenix-v2-predictor",
        "release_arm": "ptxv2",
        "stage_id": "cofold-rescore-protenix-v2",
    },
)

THREE_TO_ONE = {
    "ALA": "A",
    "ARG": "R",
    "ASN": "N",
    "ASP": "D",
    "CYS": "C",
    "GLN": "Q",
    "GLU": "E",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LEU": "L",
    "LYS": "K",
    "MET": "M",
    "PHE": "F",
    "PRO": "P",
    "SER": "S",
    "THR": "T",
    "TRP": "W",
    "TYR": "Y",
    "VAL": "V",
}


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
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


def load_json(path: Path) -> Any:
    return json.loads(path.read_text())


def load_executor_measurement_fields() -> list[str]:
    """Read MEASUREMENT_SOURCE_FIELDS from the executor source itself."""
    executor_path = PACKAGE_ROOT / "lane.py"
    source = executor_path.read_text()
    match = re.search(
        r"^MEASUREMENT_SOURCE_FIELDS\s*=\s*\((.*?)^\)",
        source,
        re.MULTILINE | re.DOTALL,
    )
    if match is None:
        raise RuntimeError(f"MEASUREMENT_SOURCE_FIELDS was not found in {executor_path}")
    fields = re.findall(r"['\"]([a-zA-Z0-9_]+)['\"]", match.group(1))
    if not fields:
        raise RuntimeError(f"MEASUREMENT_SOURCE_FIELDS was empty in {executor_path}")
    if len(fields) != len(set(fields)):
        raise RuntimeError("MEASUREMENT_SOURCE_FIELDS contains duplicate names")
    return fields


def load_site_metric_basis() -> str:
    schema = load_json(OBSERVATION_SCHEMA_PATH)
    values = schema["properties"]["site_metric_basis"]["enum"]
    if not values:
        raise RuntimeError("site_metric_basis has no registered enum value")
    return str(values[-1])


def load_profile_adapters(profile_path: Path) -> dict[str, dict[str, Any]]:
    """Merge the profile base and overlay with the repository profile rules."""
    profile = load_json(profile_path)
    merged: dict[str, dict[str, Any]] = {}
    base_name = profile.get("base_profile")
    if base_name:
        base_path = profile_path.parent / str(base_name)
        base = load_json(base_path)
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


def render_argv(template: list[str], context: dict[str, Any]) -> list[str]:
    rendered: list[str] = []
    for item in template:
        if item == UNRESOLVED:
            raise RuntimeError("profile command contains unresolved argv value")
        missing = [token for token in TOKEN_RE.findall(item) if token not in context]
        if missing:
            raise RuntimeError(f"profile command contains unresolved tokens: {missing}")
        rendered.append(TOKEN_RE.sub(lambda match: str(context[match.group(1)]), item))
    return rendered


def release_root(index: dict[str, Any]) -> str:
    cached = index.get("_derived_release_root")
    if isinstance(cached, str) and cached:
        return cached
    names = list(index["entries"])
    roots = {
        name.split("/designs/", 1)[0]
        for name in names
        if "/designs/" in name
    }
    if len(roots) != 1:
        raise RuntimeError(f"could not identify one release root from {len(roots)} roots")
    root = next(iter(roots))
    index["_derived_release_root"] = root
    return root


def load_local_central_directory(cache_path: Path) -> dict[str, Any] | None:
    """Convert a cached copy of the release central directory into the fetcher cache shape.

    The copy sits beside the index cache so that pointing --cache-dir at a
    different tree moves both together.
    """
    local_cdir = cache_path.parent / CENTRAL_DIRECTORY_CACHE_NAME
    if not local_cdir.is_file():
        return None
    try:
        parsed = remote_zip.parse_central_directory(
            local_cdir.read_bytes(),
            expected_entries=0,
        )
    except Exception:
        return None
    if not parsed:
        return None
    payload = {
        "url": RELEASE_URL,
        "file_size": ARCHIVE_SIZE,
        "zip64": True,
        "cd_offset": None,
        "cd_size": local_cdir.stat().st_size,
        "entry_count": len(parsed),
        "entries": parsed,
        "source": "local central directory cache",
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(payload, sort_keys=True))
    return payload


def load_release_index(
    cache_path: Path,
    counter: remote_zip.TransferCounter,
) -> tuple[remote_zip.RemoteFile, dict[str, Any], str]:
    remote = remote_zip.RemoteFile(RELEASE_URL, counter)
    if cache_path.is_file():
        try:
            cached = load_json(cache_path)
        except json.JSONDecodeError:
            cached = None
        if isinstance(cached, dict) and cached.get("url") == RELEASE_URL:
            remote.seed_size(int(cached["file_size"]))
            return remote, cached, "cached index"

    local = load_local_central_directory(cache_path)
    if local is not None:
        remote.seed_size(int(local["file_size"]))
        return remote, local, "local cdir.bin"

    index = remote_zip.load_index(remote, str(cache_path), max_cd_bytes=200 * 1024 * 1024)
    return remote, index, "HTTP range index"


def member_path(index: dict[str, Any], target: str, design: str, arm: str, seed: int, leaf: str) -> str:
    root = release_root(index)
    exact = f"{root}/designs/{target}/{design}/{arm}/seed_{seed}/{leaf}"
    if exact in index["entries"]:
        return exact
    raise KeyError(exact)


def find_unique_suffix(index: dict[str, Any], suffix: str) -> str:
    matches = [name for name in index["entries"] if name.endswith(suffix)]
    if len(matches) != 1:
        raise RuntimeError(f"expected one release member ending {suffix!r}, found {len(matches)}")
    return matches[0]


def cached_member_path(cache_dir: Path, name: str) -> Path:
    safe = name.replace("/", "__")
    return cache_dir / "members" / safe


def fetch_member(
    remote: remote_zip.RemoteFile,
    index: dict[str, Any],
    cache_dir: Path,
    name: str,
) -> Path:
    destination = cached_member_path(cache_dir, name)
    if destination.is_file():
        return destination
    entry = index["entries"].get(name)
    if entry is None:
        raise KeyError(name)
    payload = remote_zip.extract_member(remote, entry, name)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".partial")
    temporary.write_bytes(payload)
    os.replace(temporary, destination)
    return destination


def fetch_members(
    remote: remote_zip.RemoteFile,
    index: dict[str, Any],
    cache_dir: Path,
    names: list[str],
    workers: int,
) -> dict[str, Path]:
    unique = list(dict.fromkeys(names))
    result: dict[str, Path] = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        futures = {
            pool.submit(fetch_member, remote, index, cache_dir, name): name for name in unique
        }
        for future in concurrent.futures.as_completed(futures):
            name = futures[future]
            result[name] = future.result()
    return result


def read_summary_rows(summary_path: Path, target: str) -> tuple[list[dict[str, str]], list[str]]:
    if not summary_path.is_file():
        raise FileNotFoundError(summary_path)
    with summary_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise RuntimeError(f"{summary_path} has no header")
        rows = [row for row in reader if row.get("target") == target]
        return rows, list(reader.fieldnames)


def choose_designs(
    rows: list[dict[str, str]],
    requested: list[str],
    limit: int,
) -> list[dict[str, str]]:
    by_name = {row.get("full_name", ""): row for row in rows}
    if requested:
        missing = [name for name in requested if name not in by_name]
        if missing:
            raise RuntimeError("requested designs are absent from design_summary.csv: " + ", ".join(missing))
        return [by_name[name] for name in requested]
    return rows[:limit]


def column_maps(header: list[str]) -> tuple[dict[str, str], dict[str, str]]:
    ipsae = {name.removeprefix("ipsae_min_"): name for name in header if name.startswith("ipsae_min_")}
    dockq = {name.removeprefix("sc_dockq_"): name for name in header if name.startswith("sc_dockq_")}
    for spec in REQUIRED_ADAPTERS:
        if spec["release_arm"] not in ipsae or spec["release_arm"] not in dockq:
            raise RuntimeError(
                f"design_summary.csv lacks published columns for {spec['release_arm']}; "
                f"ipSAE arms={sorted(ipsae)}, sc_DockQ arms={sorted(dockq)}"
            )
    return ipsae, dockq


def parse_published_epitope(value: str, target_chain_id: str) -> list[str]:
    labels: list[str] = []
    pattern = re.compile(r"^\s*([A-Za-z0-9_]+)\s*:\s*[A-Za-z]{1,3}(-?\d+)\s*$")
    for entry in value.split(";"):
        match = pattern.match(entry)
        if match:
            labels.append(f"{target_chain_id}:{int(match.group(2))}")
    if not labels:
        raise RuntimeError("selected design has no parseable epitope_residues")
    return list(dict.fromkeys(labels))


def chain_sequence(structure: binder_metrics.Structure, chain_id: str) -> str:
    return "".join(THREE_TO_ONE.get((residue.comp_id or "").upper(), "X") for residue in structure.chain_residues(chain_id))


def target_sequence_from_release(model_path: Path, binder_sequence: str) -> tuple[str, str, str, dict[str, str]]:
    structure = binder_metrics.parse_cif_atoms(model_path, argument="released model.cif")
    sequences = {chain: chain_sequence(structure, chain) for chain in structure.chain_ids()}
    binder_chains = [chain for chain, sequence in sequences.items() if sequence == binder_sequence]
    if len(binder_chains) != 1:
        raise RuntimeError(
            f"released model {model_path.name} does not have one chain matching the ordered binder "
            f"sequence; matches={binder_chains}, chain lengths={ {k: len(v) for k, v in sequences.items()} }"
        )
    binder_chain = binder_chains[0]
    target_chains = [chain for chain in sequences if chain != binder_chain]
    if len(target_chains) != 1:
        raise RuntimeError(f"expected one target chain beside binder {binder_chain}, found {target_chains}")
    target_chain = target_chains[0]
    return sequences[target_chain], target_chain, binder_chain, sequences


def read_parquet_metadata(path: Path, selected: set[str], target: str) -> dict[tuple[str, str, int], dict[str, Any]]:
    try:
        import polars as pl
    except ImportError as exc:
        raise RuntimeError("polars is required to read the released predictions.parquet") from exc
    frame = pl.read_parquet(path)
    needed = {"design_name", "target", "cofolding_model", "seed", "iptm_pae", "ptm_pae", "plddt_binder", "plddt_target"}
    missing = sorted(needed - set(frame.columns))
    if missing:
        raise RuntimeError(f"released predictions.parquet lacks columns: {missing}")
    result: dict[tuple[str, str, int], dict[str, Any]] = {}
    for row in frame.filter((pl.col("target") == target) & pl.col("design_name").is_in(sorted(selected))).iter_rows(named=True):
        arm = str(row["cofolding_model"])
        try:
            seed = int(str(row["seed"]))
        except (TypeError, ValueError):
            continue
        plddt_values = [float(value) for value in (row.get("plddt_binder"), row.get("plddt_target")) if value is not None]
        mean_plddt = sum(plddt_values) / len(plddt_values) if plddt_values else None
        result[(str(row["design_name"]), arm, seed)] = {
            "iptm": float(row["iptm_pae"]) if row.get("iptm_pae") is not None else None,
            "ptm": float(row["ptm_pae"]) if row.get("ptm_pae") is not None else None,
            "mean_plddt": mean_plddt,
            "published_parquet_ipsae": float(row["ipsae_min"]) if row.get("ipsae_min") is not None else None,
            "published_parquet_sc_dockq": float(row["sc_dockq"]) if row.get("sc_dockq") is not None else None,
        }
    return result


def write_fake_sources(shim_dir: Path) -> tuple[Path, Path]:
    shim_dir.mkdir(parents=True, exist_ok=True)
    sitecustomize = shim_dir / "sitecustomize.py"
    sitecustomize.write_text(
        textwrap.dedent(
            r'''
            import hashlib
            import json
            import os
            import sys
            import types

            import numpy


            def _cases():
                with open(os.environ["REPLAY_CASES_JSON"], encoding="utf-8") as handle:
                    return json.load(handle)["cases"]


            def _log(value):
                path = os.environ.get("REPLAY_FAKE_LOG")
                if path:
                    os.makedirs(os.path.dirname(path), exist_ok=True)
                    with open(path, "a", encoding="utf-8") as handle:
                        handle.write(json.dumps(value, sort_keys=True) + "\n")


            if os.environ.get("REPLAY_ESM_STUB") == "1":
                class _FakeLinalg:
                    @staticmethod
                    def svd(*args, **kwargs):
                        raise RuntimeError("replay stub SVD was reached")


                torch = types.ModuleType("torch")
                torch.linalg = _FakeLinalg()
                torch.isfinite = numpy.isfinite
                torch.nan_to_num = numpy.nan_to_num
                sys.modules["torch"] = torch

                class _FakeModel:
                    def __init__(self, checkpoint):
                        self.checkpoint = checkpoint
                        self.device = None
                        self.kernel_backend = None
                        self.chunk_size = None

                    @classmethod
                    def from_pretrained(cls, checkpoint):
                        _log({"call": "from_pretrained", "checkpoint": checkpoint})
                        return cls(checkpoint)

                    def to(self, device):
                        self.device = device
                        return self

                    def eval(self):
                        return self

                    def set_kernel_backend(self, value):
                        self.kernel_backend = value

                    def set_chunk_size(self, value):
                        self.chunk_size = value


                transformers = types.ModuleType("transformers")
                transformers_models = types.ModuleType("transformers.models")
                transformers_esm = types.ModuleType("transformers.models.esmfold2")
                modeling = types.ModuleType("transformers.models.esmfold2.modeling_esmfold2")
                modeling.ESMFold2Model = _FakeModel
                transformers_esm.modeling_esmfold2 = modeling
                transformers_models.esmfold2 = transformers_esm
                transformers.models = transformers_models
                sys.modules["transformers"] = transformers
                sys.modules["transformers.models"] = transformers_models
                sys.modules["transformers.models.esmfold2"] = transformers_esm
                sys.modules["transformers.models.esmfold2.modeling_esmfold2"] = modeling

                class _MSA:
                    def __init__(self, path):
                        self.path = path

                    @classmethod
                    def from_a3m(cls, path, max_sequences):
                        _log({"call": "msa_from_a3m", "path": path, "max_sequences": max_sequences})
                        return cls(path)


                esm = types.ModuleType("esm")
                esm_models = types.ModuleType("esm.models")
                esm_esmfold2 = types.ModuleType("esm.models.esmfold2")
                esm_utils = types.ModuleType("esm.utils")
                esm_utils_msa = types.ModuleType("esm.utils.msa")
                esm_utils_msa_msa = types.ModuleType("esm.utils.msa.msa")
                esm_utils_msa_msa.MSA = _MSA

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
                        self.text = text

                    def to_mmcif(self):
                        return self.text


                # `fold` returns one MolecularComplexResult at one diffusion
                # sample and a list otherwise. Source: local return-shape fixture.
                class _MolecularComplexResult:
                    pass


                class _Prediction(_MolecularComplexResult):
                    def __init__(self, record, pae):
                        self.complex = _Complex(open(record["model_path"], encoding="utf-8").read())
                        self.pae = pae
                        self.iptm = record["iptm"]
                        self.ptm = record["ptm"]
                        self.plddt = numpy.full(pae.shape[0], record["mean_plddt"] or 0.0, dtype=float)


                class _Builder:
                    def fold(self, model, spi, num_loops, num_sampling_steps, num_diffusion_samples, seed):
                        target, binder = spi.sequences
                        arm = os.environ["REPLAY_RELEASE_ARM"]
                        key = arm + "|" + hashlib.sha256(binder.sequence.encode()).hexdigest() + "|" + str(seed)
                        record = _cases()[key]
                        pae = numpy.load(record["pae_path"], allow_pickle=True)["pae"].astype(float)
                        _log({
                            "call": "fold",
                            "arm": arm,
                            "seed": int(seed),
                            "target_chain": target.id,
                            "binder_chain": binder.id,
                            "target_length": len(target.sequence),
                            "binder_length": len(binder.sequence),
                            "target_msa_present": target.msa is not None,
                            "checkpoint": model.checkpoint,
                            "kernel_backend": model.kernel_backend,
                            "chunk_size": model.chunk_size,
                            "num_loops": num_loops,
                            "num_sampling_steps": num_sampling_steps,
                            "num_diffusion_samples": num_diffusion_samples,
                        })
                        # Match the local fixture at the one-sample boundary.
                        predictions = [_Prediction(record, pae) for _ in range(num_diffusion_samples)]
                        return predictions[0] if num_diffusion_samples == 1 else predictions


                esm_esmfold2.ESMFold2InputBuilder = _Builder
                esm_esmfold2.ProteinInput = _ProteinInput
                esm_esmfold2.StructurePredictionInput = _StructurePredictionInput
                esm_models.esmfold2 = esm_esmfold2
                esm.models = esm_models
                esm_utils_msa.msa = esm_utils_msa_msa
                esm_utils.msa = esm_utils_msa
                # `fold` returns this class at one sample. Source: local return-shape fixture.
                # The adapter imports it to check the documented return type.
                esm_utils_structure = types.ModuleType("esm.utils.structure")
                esm_utils_molecular_complex = types.ModuleType("esm.utils.structure.molecular_complex")
                esm_utils_molecular_complex.MolecularComplexResult = _MolecularComplexResult
                esm_utils_structure.molecular_complex = esm_utils_molecular_complex
                esm_utils.structure = esm_utils_structure
                esm_utils_msa_msa.__package__ = "esm.utils.msa"
                esm.utils = esm_utils
                sys.modules["esm"] = esm
                sys.modules["esm.models"] = esm_models
                sys.modules["esm.models.esmfold2"] = esm_esmfold2
                sys.modules["esm.utils"] = esm_utils
                sys.modules["esm.utils.msa"] = esm_utils_msa
                sys.modules["esm.utils.msa.msa"] = esm_utils_msa_msa
                sys.modules["esm.utils.structure"] = esm_utils_structure
                sys.modules["esm.utils.structure.molecular_complex"] = esm_utils_molecular_complex
            ''').lstrip())

    protenix = shim_dir / "bin" / "protenix"
    protenix.parent.mkdir(parents=True, exist_ok=True)
    protenix.write_text(
        textwrap.dedent(
            r'''
            #!/usr/bin/env python3
            import hashlib
            import json
            import os
            import shutil
            import sys

            import numpy


            def value(flag):
                index = sys.argv.index(flag)
                return sys.argv[index + 1]


            if len(sys.argv) < 2 or sys.argv[1] != "pred":
                raise SystemExit("replay protenix stub accepts only pred")
            with open(os.environ["REPLAY_CASES_JSON"], encoding="utf-8") as handle:
                cases = json.load(handle)["cases"]
            with open(value("--input"), encoding="utf-8") as handle:
                document = json.load(handle)
            name = document[0]["name"]
            binder_sequence = document[0]["sequences"][1]["proteinChain"]["sequence"]
            arm = os.environ["REPLAY_RELEASE_ARM"]
            out_dir = value("--out_dir")
            seeds = [int(item) for item in value("--seeds").split(",") if item]
            for seed in seeds:
                key = arm + "|" + hashlib.sha256(binder_sequence.encode()).hexdigest() + "|" + str(seed)
                record = cases[key]
                seed_dir = os.path.join(out_dir, name, "seed_" + str(seed))
                os.makedirs(seed_dir, exist_ok=True)
                shutil.copyfile(record["model_path"], os.path.join(seed_dir, name + "_sample_0.cif"))
                pae = numpy.load(record["pae_path"], allow_pickle=True)["pae"].astype(float).tolist()
                with open(os.path.join(seed_dir, name + "_full_data_sample_0.json"), "w", encoding="utf-8") as handle:
                    json.dump({"token_pair_pae": pae}, handle)
                with open(os.path.join(seed_dir, name + "_summary_confidence_sample_0.json"), "w", encoding="utf-8") as handle:
                    json.dump({
                        "iptm": record["iptm"],
                        "ptm": record["ptm"],
                        "plddt": record["mean_plddt"],
                    }, handle)
            log_path = os.environ.get("REPLAY_FAKE_LOG")
            if log_path:
                with open(log_path, "a", encoding="utf-8") as handle:
                    handle.write(json.dumps({"call": "protenix_pred", "arm": arm, "name": name, "seeds": seeds}, sort_keys=True) + "\n")
            ''').lstrip())
    protenix.chmod(0o755)
    return sitecustomize, protenix


def build_release_cases(
    selected: list[dict[str, str]],
    target: str,
    index: dict[str, Any],
    remote: remote_zip.RemoteFile,
    cache_dir: Path,
    metadata: dict[tuple[str, str, int], dict[str, Any]],
    workers: int,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    names: list[str] = []
    requested: list[tuple[dict[str, str], str, int, str, str]] = []
    missing_counts = {spec["release_arm"]: 0 for spec in REQUIRED_ADAPTERS}
    for row in selected:
        design = str(row["full_name"])
        designed_name = f"{release_root(index)}/designs/{target}/{design}/designed/designed.cif"
        if designed_name in index["entries"]:
            names.append(designed_name)
        for spec in REQUIRED_ADAPTERS:
            for seed in range(5):
                for leaf in ("model.cif", "pae.npz"):
                    try:
                        name = member_path(index, target, design, spec["release_arm"], seed, leaf)
                    except KeyError:
                        missing_counts[spec["release_arm"]] += 1
                        continue
                    names.append(name)
                    requested.append((row, spec["release_arm"], seed, leaf, name))
    paths = fetch_members(remote, index, cache_dir, names, workers)
    cases: list[dict[str, Any]] = []
    for row in selected:
        design = str(row["full_name"])
        designed_name = f"{release_root(index)}/designs/{target}/{design}/designed/designed.cif"
        designed_path = paths.get(designed_name)
        case = {"row": row, "design": design, "designed_path": designed_path, "records": {}, "skip": {}}
        if designed_path is None:
            case["skip"]["all"] = "design has no released designed/designed.cif"
            cases.append(case)
            continue
        for spec in REQUIRED_ADAPTERS:
            arm = spec["release_arm"]
            arm_records: dict[int, dict[str, Any]] = {}
            for seed in range(5):
                model_name = None
                pae_name = None
                try:
                    model_name = member_path(index, target, design, arm, seed, "model.cif")
                    pae_name = member_path(index, target, design, arm, seed, "pae.npz")
                except KeyError:
                    case["skip"][arm] = f"missing released model.cif or pae.npz for seed {seed}"
                    break
                model_path = paths.get(model_name)
                pae_path = paths.get(pae_name)
                release_meta = metadata.get((design, arm, seed))
                if model_path is None or pae_path is None or release_meta is None:
                    case["skip"][arm] = f"missing cached release member or predictions.parquet row for seed {seed}"
                    break
                arm_records[seed] = {
                    "model_path": str(model_path),
                    "pae_path": str(pae_path),
                    **release_meta,
                }
            if arm not in case["skip"]:
                case["records"][arm] = arm_records
        cases.append(case)
    return cases, missing_counts


def make_campaign(
    cases: list[dict[str, Any]],
    target_id: str,
    target_sequence: str,
    target_structure: Path,
    site_residues: list[str],
    residue_map_path: Path,
    profile_adapters: dict[str, dict[str, Any]],
    fields: list[str],
    run_root: Path,
    campaign_template: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]], Path, Path, Path]:
    artifact_root = run_root / "artifacts"
    inputs_root = artifact_root / "inputs"
    receipts_dir = run_root / "receipts"
    template_target = campaign_template["targets"][0]
    template_site = template_target["site"]
    target_chain_id = str(campaign_template["binder"]["target_chain_id"])
    binder_chain_id = str(campaign_template["binder"]["binder_chain_id"])
    msa_path = inputs_root / "msa" / f"{target_id}.a3m"
    target_sequence_path = inputs_root / "targets" / f"{target_id}.fasta"
    target_sequence_path.parent.mkdir(parents=True, exist_ok=True)
    target_sequence_path.write_text(f">{target_id}\n{target_sequence}\n")
    msa_path.parent.mkdir(parents=True, exist_ok=True)
    msa_path.write_text(f">{target_id}\n{target_sequence}\n")
    receipts_dir.mkdir(parents=True, exist_ok=True)

    candidates: list[dict[str, Any]] = []
    for case in cases:
        row = case["row"]
        if case.get("skip", {}).get("all") or case.get("designed_path") is None:
            continue
        candidate_id = str(row["full_name"])
        candidate_dir = inputs_root / "candidates" / candidate_id
        candidate_dir.mkdir(parents=True, exist_ok=True)
        fasta_path = candidate_dir / "binder.fasta"
        fasta_path.write_text(f">{candidate_id}\n{row['sequence']}\n")
        candidates.append(
            {
                "candidate_id": candidate_id,
                "sequence_path": str(fasta_path),
                "sequence_sha256": sha256_file(fasta_path),
                "design_pose_path": str(Path(case["designed_path"]).resolve()),
                "design_pose_sha256": sha256_file(Path(case["designed_path"])),
                "origin_generator": str(row.get("generator") or "released-design-summary"),
            }
        )
    write_jsonl(artifact_root / "filters" / "passing-candidates.jsonl", candidates)
    write_jsonl(artifact_root / "optimization" / "rescore-candidates.jsonl", candidates)

    predictors = [
        {
            "id": spec["predictor_id"],
            "adapter_id": spec["adapter_id"],
            "enabled": True,
            "screen_stage": "cofold-screen-" + spec["predictor_id"],
            "rescore_stage": spec["stage_id"],
        }
        for spec in REQUIRED_ADAPTERS
    ]
    stages = []
    for spec in REQUIRED_ADAPTERS:
        stages.append(
            {
                "stage_id": spec["stage_id"],
                "adapter_id": spec["adapter_id"],
                "mode": "smoke_scale",
                "outputs": [
                    {
                        "artifact_id": spec["predictor_id"] + "-rescore",
                        "artifact_type": "raw-prediction-manifest",
                        "kind": "jsonl",
                        "path_template": "{{attempt_dir}}/{{phase}}/cofold-observations.jsonl",
                    }
                ],
            }
        )
    metric_basis = load_site_metric_basis()
    config = {
        "schema_version": 1,
        "campaign_id": "replay-released-adapters",
        "run_id": "replay",
        "profile": {"profile_id": "replay-adapters"},
        "binder": {
            "minimum_length": min(len(str(case["row"]["sequence"])) for case in cases if not case.get("skip", {}).get("all")),
            "maximum_length": max(len(str(case["row"]["sequence"])) for case in cases if not case.get("skip", {}).get("all")),
            "target_chain_id": target_chain_id,
            "binder_chain_id": binder_chain_id,
        },
        "targets": [
            {
                "target_id": target_id,
                "role": "primary",
                "source_id": "released-design-summary",
                "entities": [{"entity_id": "target-protein", "type": "protein", "chain_ids": [target_chain_id], "required": True}],
                "structure_path": str(target_structure.resolve()),
                "structure_sha256": sha256_file(target_structure),
                "chains": [{"chain_id": target_chain_id, "role": "design-target"}],
                "site": {
                    "mode": template_site["mode"],
                    "design_residues": site_residues,
                    "reference_contact_residues": site_residues,
                    "contact_cutoff_angstrom": template_site["contact_cutoff_angstrom"],
                    "atom_selection": template_site["atom_selection"],
                    "residue_map_path": str(residue_map_path.resolve()),
                    "runtime_residue_map_path": str(residue_map_path.resolve()),
                    "residue_map_sha256": sha256_file(residue_map_path),
                    "hotspot_source": "explicit",
                    "hotspot_residues": site_residues,
                },
                "state_label": "primary",
            }
        ],
        "controls": {"positive": [], "negative": []},
        "cofold": {"predictors": predictors, "screen_seeds": [0], "rescore_seeds": [0, 1, 2, 3, 4]},
        "adapters": [
            {"adapter_id": spec["adapter_id"], "model_revision": str(profile_adapters[spec["adapter_id"]].get("model_revision", UNRESOLVED))}
            for spec in REQUIRED_ADAPTERS
        ],
        "scoring": {
            "implementations": {
                "ipsae_revision": binder_metrics.IPSAE_IMPLEMENTATION_REVISION,
                "ipsae_interface_cutoff_angstrom": binder_metrics.IPSAE_INTERFACE_CUTOFF_ANGSTROM,
                "dockq_revision": binder_metrics.DOCKQ_IMPLEMENTATION_REVISION,
                "site_scorer_revision": binder_metrics.SITE_SCORER_IMPLEMENTATION_REVISION,
                "site_metric_basis": metric_basis,
            },
            "required_observation_fields": fields,
            "seed_reduction": "maximum-ipsae-min-per-predictor",
            "paired_pose_rule": "use-sc-dockq-from-the-same-winning-seed",
        },
        "stages": stages,
    }
    config_path = write_json(run_root / "config.resolved.json", config)
    plan_path = write_json(run_root / "plan.json", {"schema_version": 1})
    return config, candidates, config_path, plan_path, msa_path


def command_context(
    config_path: Path,
    plan_path: Path,
    artifact_root: Path,
    receipts_dir: Path,
    run_root: Path,
    stage_id: str,
    count: int,
) -> dict[str, Any]:
    attempt_dir = artifact_root / "stages" / stage_id / "attempts" / "replay"
    return {
        "stage_id": stage_id,
        "phase": "scale",
        "count": count,
        "attempt_id": "replay",
        "attempt_dir": str(attempt_dir),
        "receipts_dir": str(receipts_dir),
        "artifact_root": str(artifact_root),
        "run_root": str(run_root),
        "config_path": str(config_path),
        "plan_path": str(plan_path),
        "optimization_round": 0,
        "target_id": "PD-L1",
    }


def run_process(argv: list[str], env: dict[str, str], log_path: Path) -> subprocess.CompletedProcess[str]:
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
        + f"returncode: {completed.returncode}\n"
        + "--- stdout ---\n"
        + completed.stdout
        + "\n--- stderr ---\n"
        + completed.stderr
    )
    return completed


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows = []
    for line in path.read_text().splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def make_environment(shim_dir: Path, cases_json: Path, release_arm: str, log_path: Path) -> dict[str, str]:
    env = dict(os.environ)
    env["PYTHONSAFEPATH"] = "1"
    env["PYTHONPATH"] = str(shim_dir)
    env["PATH"] = str(shim_dir / "bin") + os.pathsep + env.get("PATH", "")
    env["REPLAY_CASES_JSON"] = str(cases_json)
    env["REPLAY_RELEASE_ARM"] = release_arm
    env["REPLAY_FAKE_LOG"] = str(log_path)
    env["REPLAY_ESM_STUB"] = "1" if release_arm in {"ef2fast", "ef2full"} else "0"
    return env


def supplement_command(argv: list[str], target_id: str, target_fasta: Path, site_residues: list[str]) -> list[str]:
    result = list(argv)
    if "--target-sequence" not in result:
        result.extend(["--target-sequence", f"{target_id}={target_fasta}"])
    if "--hotspot-residues" not in result:
        result.extend(["--hotspot-residues", f"{target_id}={','.join(site_residues)}"])
    return result


def verify_contract_rows(
    rows: list[dict[str, Any]],
    measurement_fields: list[str],
    expected_count: int,
) -> dict[str, Any]:
    scored = [row for row in rows if row.get("status") == "scored"]
    failed = [row for row in rows if row.get("status") == "failed"]
    missing_fields: list[str] = []
    null_fields: list[str] = []
    artifact_fields: list[str] = []
    hash_mismatches: list[str] = []
    artifact_count = 0
    for row in scored:
        label = f"{row.get('candidate_id')} seed {row.get('seed')}"
        required_artifacts = ("predicted_complex_path", "pae_path", "metric_source_path")
        missing_artifacts = [name for name in required_artifacts if not row.get(name)]
        if missing_artifacts:
            artifact_fields.extend(f"{label}: {name}" for name in missing_artifacts)
            continue
        metric_path = Path(str(row["metric_source_path"]))
        if not metric_path.is_file():
            artifact_fields.append(f"{label}: metric file")
            continue
        document = load_json(metric_path)
        measurement = document.get("measurement", {})
        absent = [field for field in measurement_fields if field not in measurement]
        null = [field for field in measurement_fields if measurement.get(field) is None]
        missing_fields.extend(f"{label}: {field}" for field in absent)
        null_fields.extend(f"{label}: {field}" for field in null)
        complex_path = Path(str(row["predicted_complex_path"]))
        pae_path = Path(str(row["pae_path"]))
        for field, path in (("predicted_complex_sha256", complex_path), ("pae_sha256", pae_path), ("metric_source_sha256", metric_path)):
            if not path.is_file():
                artifact_fields.append(f"{label}: {path}")
            elif row.get(field) != sha256_file(path):
                hash_mismatches.append(f"{label}: {field}")
        if all(path.is_file() for path in (complex_path, pae_path, metric_path)):
            artifact_count += 1
    return {
        "expected_rows": expected_count,
        "actual_rows": len(rows),
        "scored_rows": len(scored),
        "failed_rows": len(failed),
        "all_rows_present": len(rows) == expected_count,
        "all_scored": len(scored) == expected_count,
        "artifact_rows": artifact_count,
        "missing_artifact_fields": artifact_fields,
        "hash_mismatches": hash_mismatches,
        "missing_measurement_fields": missing_fields,
        "null_measurement_fields": null_fields,
        "contract_ok": (
            len(rows) == expected_count
            and len(scored) == expected_count
            and not artifact_fields
            and not hash_mismatches
            and not missing_fields
            and not null_fields
            and artifact_count == expected_count
        ),
    }


def pearson(left: list[float], right: list[float]) -> float | None:
    if len(left) < 2 or len(left) != len(right):
        return None
    mean_left = sum(left) / len(left)
    mean_right = sum(right) / len(right)
    numerator = sum((a - mean_left) * (b - mean_right) for a, b in zip(left, right))
    denom_left = math.sqrt(sum((a - mean_left) ** 2 for a in left))
    denom_right = math.sqrt(sum((b - mean_right) ** 2 for b in right))
    if denom_left == 0 or denom_right == 0:
        return None
    return numerator / (denom_left * denom_right)


def metric_summary(pairs: list[tuple[float, float]]) -> dict[str, Any]:
    if not pairs:
        return {"n": 0, "pearson_r": None, "mean_abs_diff": None, "max_abs_diff": None}
    diffs = [abs(left - right) for left, right in pairs]
    return {
        "n": len(pairs),
        "pearson_r": pearson([left for left, _ in pairs], [right for _, right in pairs]),
        "mean_abs_diff": sum(diffs) / len(diffs),
        "max_abs_diff": max(diffs),
    }


def compare_results(
    cases: list[dict[str, Any]],
    manifest_rows: dict[str, list[dict[str, Any]]],
    ipsae_columns: dict[str, str],
    dockq_columns: dict[str, str],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    output_rows: list[dict[str, Any]] = []
    summaries: dict[str, Any] = {}
    case_by_id = {str(case["row"]["full_name"]): case for case in cases}
    for spec in REQUIRED_ADAPTERS:
        arm = spec["release_arm"]
        rows = manifest_rows.get(arm, [])
        by_candidate: dict[str, list[dict[str, Any]]] = {}
        for row in rows:
            by_candidate.setdefault(str(row.get("candidate_id")), []).append(row)
        ipsae_pairs: list[tuple[float, float]] = []
        dockq_pairs: list[tuple[float, float]] = []
        for case in cases:
            row = case["row"]
            design = str(row["full_name"])
            if arm not in case.get("records", {}):
                output_rows.append({
                    "design_name": design,
                    "target": row.get("target"),
                    "arm": arm,
                    "status": "skipped",
                    "skip_reason": case.get("skip", {}).get(arm, case.get("skip", {}).get("all", "no released structure")),
                })
                continue
            adapter_rows = sorted(by_candidate.get(design, []), key=lambda item: int(item.get("seed", 0)))
            scored_rows: list[dict[str, Any]] = []
            for item in adapter_rows:
                if item.get("status") != "scored":
                    continue
                metric_path = item.get("metric_source_path")
                if not metric_path or not Path(str(metric_path)).is_file():
                    continue
                measurement = load_json(Path(str(metric_path))).get("measurement", {})
                enriched = dict(item)
                enriched["ipsae_min"] = measurement.get("ipsae_min")
                enriched["sc_dockq"] = measurement.get("sc_dockq")
                scored_rows.append(enriched)
            if not scored_rows:
                output_rows.append({
                    "design_name": design,
                    "target": row.get("target"),
                    "arm": arm,
                    "status": "adapter_failed",
                    "adapter_row_count": len(adapter_rows),
                })
                continue
            winner = max(scored_rows, key=lambda item: float(item["ipsae_min"]))
            replay_ipsae = float(winner["ipsae_min"])
            replay_dockq = float(winner["sc_dockq"]) if winner.get("sc_dockq") is not None else None
            published_ipsae = float(row[ipsae_columns[arm]]) if row.get(ipsae_columns[arm]) not in (None, "") else None
            published_dockq = float(row[dockq_columns[arm]]) if row.get(dockq_columns[arm]) not in (None, "") else None
            if published_ipsae is not None:
                ipsae_pairs.append((published_ipsae, replay_ipsae))
            if published_dockq is not None and replay_dockq is not None:
                dockq_pairs.append((published_dockq, replay_dockq))
            output_rows.append({
                "design_name": design,
                "target": row.get("target"),
                "arm": arm,
                "status": "compared",
                "adapter_row_count": len(adapter_rows),
                "adapter_scored_rows": len(scored_rows),
                "winning_seed": winner.get("seed"),
                "published_ipsae_min": published_ipsae,
                "replayed_ipsae_min": replay_ipsae,
                "ipsae_abs_diff": abs(published_ipsae - replay_ipsae) if published_ipsae is not None else None,
                "published_sc_dockq": published_dockq,
                "replayed_sc_dockq": replay_dockq,
                "sc_dockq_abs_diff": abs(published_dockq - replay_dockq) if published_dockq is not None and replay_dockq is not None else None,
                "measurement_field_count": len(load_executor_measurement_fields()),
                "contract_fields_ok": all(
                    verify_contract_rows([item], load_executor_measurement_fields(), 1)["contract_ok"] for item in scored_rows
                ),
            })
        summaries[arm] = {
            "compared_designs": len(ipsae_pairs),
            "ipsae_min": metric_summary(ipsae_pairs),
            "sc_dockq": metric_summary(dockq_pairs),
            "skipped_designs": sum(1 for item in output_rows if item.get("arm") == arm and item.get("status") == "skipped"),
            "adapter_failed_designs": sum(1 for item in output_rows if item.get("arm") == arm and item.get("status") == "adapter_failed"),
            "rows_attempted": len(rows),
            "rows_scored": sum(1 for item in rows if item.get("status") == "scored"),
            "rows_failed": sum(1 for item in rows if item.get("status") == "failed"),
            "failure_codes": dict(sorted(
                Counter(
                    str(item.get("failure_code"))
                    for item in rows
                    if item.get("status") == "failed"
                ).items()
            )),
        }
    return output_rows, summaries


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def run(args: argparse.Namespace) -> dict[str, Any]:
    cache_dir = Path(args.cache_dir).resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    summary_path = Path(args.summary_path).expanduser().resolve()
    if not summary_path.is_file():
        raise FileNotFoundError(summary_path)
    index_cache = Path(args.index_cache).resolve() if args.index_cache else cache_dir / "release-index.json"
    counter = remote_zip.TransferCounter()
    remote, index, index_source = load_release_index(index_cache, counter)
    selected_rows, header = read_summary_rows(summary_path, args.target)
    selected = choose_designs(selected_rows, args.design, args.limit)
    ipsae_columns, dockq_columns = column_maps(header)
    selected_names = {str(row["full_name"]) for row in selected}

    predictions_path = (
        Path(args.predictions_parquet).resolve()
        if args.predictions_parquet
        else cache_dir / PREDICTIONS_CACHE_NAME
    )
    predictions_source = "local predictions.parquet"
    if not predictions_path.is_file():
        predictions_member = find_unique_suffix(index, "/predictions.parquet")
        predictions_path = fetch_member(remote, index, cache_dir, predictions_member)
        predictions_source = "range-fetched predictions.parquet"
    metadata = read_parquet_metadata(predictions_path, selected_names, args.target)
    cases, missing_counts = build_release_cases(
        selected, args.target, index, remote, cache_dir, metadata, args.workers
    )

    runnable_cases = [
        case
        for case in cases
        if not case.get("skip", {}).get("all")
        and case.get("designed_path")
        and all(spec["release_arm"] in case.get("records", {}) for spec in REQUIRED_ADAPTERS)
    ]
    if not runnable_cases:
        raise RuntimeError("none of the selected designs has all three published modes and a designed.cif")
    first_model = Path(runnable_cases[0]["records"]["ef2full"][0]["model_path"])
    target_sequence, release_target_chain, release_binder_chain, release_sequences = target_sequence_from_release(
        first_model, str(runnable_cases[0]["row"]["sequence"])
    )
    campaign_template = load_json(CAMPAIGN_TEMPLATE_PATH)
    target_chain_id = str(campaign_template["binder"]["target_chain_id"])
    site_residues = parse_published_epitope(
        str(runnable_cases[0]["row"].get("epitope_residues", "")),
        target_chain_id,
    )
    residue_map_path = cache_dir / "inputs" / "residue-map.json"
    write_json(residue_map_path, {"source_to_cleaned": {label: label for label in site_residues}})

    run_root = cache_dir / "run"
    run_root.mkdir(parents=True, exist_ok=True)
    profile_adapters = load_profile_adapters(Path(args.profile).resolve())
    for spec in REQUIRED_ADAPTERS:
        if spec["adapter_id"] not in profile_adapters:
            raise RuntimeError(f"profile has no adapter record for {spec['adapter_id']}")
        if "command_argv_template" not in profile_adapters[spec["adapter_id"]]:
            raise RuntimeError(f"profile has no command_argv_template for {spec['adapter_id']}")
        if "parser_argv_template" not in profile_adapters[spec["adapter_id"]]:
            raise RuntimeError(f"profile has no parser_argv_template for {spec['adapter_id']}")
    fields = load_executor_measurement_fields()
    target_structure = Path(runnable_cases[0]["designed_path"])
    config, candidates, config_path, plan_path, msa_path = make_campaign(
        runnable_cases,
        args.target,
        target_sequence,
        target_structure,
        site_residues,
        residue_map_path,
        profile_adapters,
        fields,
        run_root,
        campaign_template,
    )
    shim_dir = cache_dir / "fakes"
    _, _ = write_fake_sources(shim_dir)
    cases_payload = {"cases": {}}
    for case in runnable_cases:
        sequence = str(case["row"]["sequence"])
        digest = hashlib.sha256(sequence.encode()).hexdigest()
        for spec in REQUIRED_ADAPTERS:
            for seed, record in case.get("records", {}).get(spec["release_arm"], {}).items():
                cases_payload["cases"][f"{spec['release_arm']}|{digest}|{seed}"] = record
    cases_json = write_json(cache_dir / "replay-cases.json", cases_payload)
    target_fasta = run_root / "artifacts" / "inputs" / "targets" / f"{args.target}.fasta"
    manifest_rows: dict[str, list[dict[str, Any]]] = {}
    process_records: list[dict[str, Any]] = []
    for spec in REQUIRED_ADAPTERS:
        template = profile_adapters[spec["adapter_id"]]["command_argv_template"]
        parser_template = profile_adapters[spec["adapter_id"]]["parser_argv_template"]
        context = command_context(
            config_path,
            plan_path,
            run_root / "artifacts",
            run_root / "receipts",
            run_root,
            spec["stage_id"],
            len(candidates),
        )
        context["target_id"] = args.target
        argv = supplement_command(render_argv(template, context), args.target, target_fasta, site_residues)
        parser_argv = render_argv(parser_template, context)
        log_dir = run_root / "logs"
        fake_log = log_dir / f"{spec['release_arm']}-fake-calls.jsonl"
        env = make_environment(shim_dir, cases_json, spec["release_arm"], fake_log)
        run_log = log_dir / f"{spec['release_arm']}-run.log"
        parse_log = log_dir / f"{spec['release_arm']}-parse.log"
        run_completed = run_process(argv, env, run_log)
        parse_completed = None
        if run_completed.returncode == 0:
            parse_completed = run_process(parser_argv, env, parse_log)
        manifest_path = Path(context["attempt_dir"]) / "scale" / "cofold-observations.jsonl"
        rows = read_jsonl(manifest_path)
        manifest_rows[spec["release_arm"]] = rows
        process_records.append(
            {
                "arm": spec["release_arm"],
                "adapter_id": spec["adapter_id"],
                "run_argv": argv,
                "parser_argv": parser_argv,
                "run_returncode": run_completed.returncode,
                "parse_returncode": parse_completed.returncode if parse_completed is not None else None,
                "run_log": str(run_log),
                "parse_log": str(parse_log),
                "fake_log": str(fake_log),
            }
        )

    contract = {
        spec["release_arm"]: verify_contract_rows(
            manifest_rows.get(spec["release_arm"], []),
            fields,
            len(candidates) * 5,
        )
        for spec in REQUIRED_ADAPTERS
    }
    comparison_rows, summaries = compare_results(cases, manifest_rows, ipsae_columns, dockq_columns)
    output_path = Path(args.output).resolve()
    write_csv(output_path, comparison_rows)
    report_data = {
        "release_url": RELEASE_URL,
        "archive_size_bytes": ARCHIVE_SIZE,
        "index_source": index_source,
        "predictions_source": predictions_source,
        "cache_dir": str(cache_dir),
        "target": args.target,
        "selected_designs": [str(row["full_name"]) for row in selected],
        "candidate_count": len(candidates),
        "missing_member_counts": missing_counts,
        "skipped_cases": {str(case["row"]["full_name"]): case.get("skip", {}) for case in cases if case.get("skip")},
        "release_chain_sequences": release_sequences,
        "release_target_chain": release_target_chain,
        "release_binder_chain": release_binder_chain,
        "campaign_chain_mapping": {
            "target": config["targets"][0]["chains"][0]["chain_id"],
            "binder": config["binder"]["binder_chain_id"],
        },
        "measurement_fields": fields,
        "measurement_field_count": len(fields),
        "processes": process_records,
        "contract": contract,
        "summaries": summaries,
        "results_csv": str(output_path),
        "transfer": counter.as_dict(),
        "config_path": str(config_path),
        "plan_path": str(plan_path),
        "cases_json": str(cases_json),
        "timestamp": time.time(),
    }
    write_json(run_root / "replay-run.json", report_data)
    return report_data


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", default="PD-L1")
    parser.add_argument("--limit", type=int, default=6)
    parser.add_argument("--design", action="append", default=[])
    # Relative, so both resolve under the working directory the reader chose.
    # An installed skill sits in a directory the platform owns and compares
    # against a manifest, so a run must not write cache or results into it.
    parser.add_argument("--cache-dir", default="replay-cache")
    parser.add_argument("--index-cache")
    parser.add_argument(
        "--predictions-parquet",
        help="local copy of the released predictions.parquet, default is "
             "predictions.parquet inside --cache-dir, range-fetched when absent",
    )
    parser.add_argument("--summary-path", type=Path, required=True)
    parser.add_argument("--profile", default=str(DEFAULT_PROFILE_PATH))
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--output", default="replay_results.csv")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        data = run(args)
    except Exception as exc:  # noqa: BLE001
        failure = {
            "error_type": type(exc).__name__,
            "error": str(exc),
            "release_url": RELEASE_URL,
            "timestamp": time.time(),
        }
        cache_dir = Path(args.cache_dir).resolve()
        cache_dir.mkdir(parents=True, exist_ok=True)
        write_json(cache_dir / "replay-failure.json", failure)
        print(f"replay_adapters failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({"results_csv": data["results_csv"], "summaries": data["summaries"], "transfer": data["transfer"]}, indent=2, sort_keys=True))
    if not any(summary["rows_scored"] for summary in data["summaries"].values()):
        print("replay_adapters produced zero scored rows", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
