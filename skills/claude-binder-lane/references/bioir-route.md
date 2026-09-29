# BioNeMo Inference Runtime (BioIR) route

BioIR is NVIDIA's Python runtime for GPU-accelerated biomolecular structure inference. Use it to refold generated binder candidates with a supported structure predictor and return structures and confidence data to the campaign. Generation stays with the selected binder generator. This guide describes a native, self-hosted handoff; Binder currently ships no BioIR adapter or prebuilt cloud deployment.

## Choose the route

- Choose BioIR when the user has an NVIDIA GPU host that meets the [installation requirements](https://docs.nvidia.com/bionemo/inference-runtime/latest/install/), wants batch structure prediction, or already has a BioIR environment. A single GPU serial run is a useful first prediction; Ray can run independent inputs on separate GPU replicas for a larger batch.
- Use `boltz-2` for a target–binder complex with protein chains and an unpaired A3M for each chain. The [support matrix](https://docs.nvidia.com/bionemo/inference-runtime/references/support-matrix/) also lists `openfold3`, `boltz-1`, AlphaFold2/OpenFold2 variants, their input coverage, and GPU support. Pick a model by required molecules and available inputs, then record the model key.
- BioIR Boltz-2 and an accelerated-kit Boltz-2 run use the same predictor family. For a campaign requiring two model families, add an independent complex predictor such as OpenFold3; ESMFold2 may additionally assess the isolated binder fold where appropriate.
- For an API endpoint, see [NVIDIA BioNeMo NIM](tool-catalogue.md#nvidia-bionemo-nim) and use its own request schema.

## Plan before compute

Identify the host and GPU, target and binder sequences, chain identities, A3M inputs, candidate count, model key, output directory, and the user's spend ceiling. Resolve routine environment choices yourself. Before starting a paid host or endpoint, obtain the user's authorization for the named provider and maximum spend; keep the run within that ceiling and stop the resource afterward. For an already authorized host, check the remaining ceiling before adding candidates or replicas.

On the GPU host, check Linux with glibc 2.34 or newer, Python 3.12, an NVIDIA GPU from the support matrix, and driver 580 or newer. Install the release wheel in an isolated Python 3.12 environment and confirm import:

```bash
python3.12 -m venv .venv
. .venv/bin/activate
python -m pip install bionemo-ir
python -c 'import bionemo_ir; print(bionemo_ir.__version__)'
nvidia-smi
```

The first model run may download checkpoints and chemical metadata. Set `BIOIR_CACHE` to persistent storage when the host is ephemeral; see [model weights and cache settings](https://docs.nvidia.com/bionemo/inference-runtime/references/model-weights/). Record the package version, model key, seed, input paths or hashes, and checkpoint identity with results. Keep credentials out of campaign files.

## Form a target–binder request

Use BioIR's documented [`InputRequest` and `build_processor` API](https://docs.nvidia.com/bionemo/inference-runtime/latest/references/api/). Each protein chain needs an unpaired A3M. Supply an existing full MSA when available; an A3M containing only the query sequence is useful for a pipeline smoke run. Keep candidate IDs stable so the returned CIF and score sidecar map to the original sequence.

```python
import hashlib
import json
from pathlib import Path

from bionemo_ir.data.schemas import InputRequest, MSARecord, Polymer
from bionemo_ir.pipeline.processor.engine_proc import EngineProcessorConfig, build_processor
from bionemo_ir.pipeline.stages.configs import FeatureGeneratorStageConfig, WriterStageConfig

candidate_id = "candidate_001"
target_sequence = "..."  # replace with the resolved target sequence
binder_sequence = "..."  # replace with this candidate's sequence
target_a3m = Path("inputs/target.a3m")
binder_a3m = Path("inputs/candidate_001.a3m")

request = InputRequest(
    input_id=candidate_id,
    polymers=[
        Polymer(polymer_type="protein", chain_id=["A"], sequence=target_sequence,
                msas=[MSARecord(path=str(target_a3m))]),
        Polymer(polymer_type="protein", chain_id=["B"], sequence=binder_sequence,
                msas=[MSARecord(path=str(binder_a3m))]),
    ],
)
config = EngineProcessorConfig(
    model_source="boltz-2",
    feature_generator_stage=FeatureGeneratorStageConfig(init_context={"random_seed": 42}),
    writer_stage=WriterStageConfig(output_path="results/bioir", format="cif"),
)
row = build_processor(config)([{"record": request, "__record_id": candidate_id}])[0]
scores = json.loads(row["scores"])
structure = Path(row["output_path"])
assert structure.is_file() and structure.stat().st_size > 0
score_path = structure.with_name(candidate_id + "_scores.json")
assert score_path.is_file() and score_path.stat().st_size > 0

def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

print(json.dumps({
    "candidate_id": candidate_id,
    "model": "boltz-2",
    "structure_path": str(structure),
    "structure_sha256": sha256(structure),
    "score_path": str(score_path),
    "score_sha256": sha256(score_path),
    "ptm": scores.get("ptm"),
    "iptm": scores.get("iptm"),
    "mean_plddt": sum(scores["plddt"]) / len(scores["plddt"]),
}, indent=2))
```

BioIR supplies model defaults unless overridden. Run one candidate against the real target first, then parse its CIF and score JSON before scaling. For throughput across independent candidates, use the [Ray replica configuration](https://docs.nvidia.com/bionemo/inference-runtime/references/ray/); size engine actors to visible GPUs. Ray assigns a complete model replica to each GPU. Preserve a per-candidate seed and model/runtime settings in the campaign manifest.

## Return evidence to the campaign

For every candidate, retain the generated sequence, target and binder chain mapping, A3M provenance, CIF/PDB path, score JSON, SHA-256 hashes of input and output artifacts, model/version, seed, and run status. Parse `row["scores"]` as JSON; it can contain pLDDT, pTM, ipTM, PAE and model-specific extras. Check the returned ID, nonempty structure, expected chains and sequences, and finite scores before ranking. Capture failures per candidate and retry after fixing the input or resource issue. Leave absent scores absent.

Import a one-off result at the next Binder boundary with its hashes and provenance. Use a supplied-candidate manifest when the source candidate also carries the sequence, design pose, target identity, residue-map identity, and lineage required by that schema. A repeatable automated dispatch calls for a separate adapter with an explicit input/output contract.

Compare the complex confidence with the other campaign filters: interface geometry, clashes, target site, design diversity, and an independent predictor when required. Report ranked structures and score definitions with the resource time and cost. A computational score is evidence for prioritization, and the agent can continue from it to the next justified campaign step.

## Primary references

- [BioIR overview and upstream repository](https://github.com/NVIDIA-BioNeMo/BioNeMo-Inference-Runtime)
- [Installation requirements](https://docs.nvidia.com/bionemo/inference-runtime/latest/install/)
- [Python API, input schema, and output rows](https://docs.nvidia.com/bionemo/inference-runtime/latest/references/api/)
- [Model and GPU support matrix](https://docs.nvidia.com/bionemo/inference-runtime/references/support-matrix/)
