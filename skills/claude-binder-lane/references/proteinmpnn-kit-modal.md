# ProteinMPNN optimization kit on Modal

Claude Science can build the shipped [recipe](../envs/proteinmpnn_kit_gpu.py)
on the scientist's connected Modal account, use it directly, or connect it to
their existing backbone-generation and prediction workflow. The recipe
preserves the [Anthropic kit](https://github.com/anthropics/uplifting-biomolecular-modeling/tree/f4f62fa6592ae4938d49b1757bea0cfeff9f468e/proteinmpnn)
at `f4f62fa6592ae4938d49b1757bea0cfeff9f468e`, ProteinMPNN at
`8907e6671bfbfc92303b5f79c4b5e6ce47cdef57`, and the complete pinned stack.
A missing campaign adapter does not prevent this direct kit route.

## Prepare the scientist's environment

Discover the connected account and the available Modal helpers. Reuse an image
and Volume only when their recorded pins match. Resolve the installed recipe
path and build the environment after the provider and spend ceiling are
authorized:

```python
result = build_env(
    "proteinmpnn_kit_gpu",
    path="<installed-skill>/envs/proteinmpnn_kit_gpu.py",
    hydrate=True,
)
```

The recipe's `build()` declares the image, `/weights` Volume, and runtime
environment. Image assembly and hydration incur provider usage when executed.
Hydration calls the upstream kit's `install --weights /weights/ProteinMPNN`,
which clones the pinned code and weights, keeps `.git` for FASTA provenance,
and checks both vanilla and soluble checkpoint digests. `MPNN_DIR` names that
clone. Pass `volume_name` to `build()` when the session needs another
account-local cache name.

The [upstream Dockerfile](https://github.com/anthropics/uplifting-biomolecular-modeling/blob/f4f62fa6592ae4938d49b1757bea0cfeff9f468e/proteinmpnn/environment/Dockerfile)
specifies CUDA 12.4.1, a hash-checked CPython 3.11.5 distribution, and the full
requirements lock with torch 2.5.1+cu124, NumPy 1.26.4, and Triton 3.1.0.
The host driver requirement is 550 or newer. Check the running driver instead
of assuming a provider's GPU label establishes compatibility. Preserve the
kit and upstream licences when redistributing the image. The recipe keeps
the release's `LICENSE` and `NOTICE` beside the kit-specific notices.

Use H100 for the default recipe. Upstream also supplies A100 and H200 card
configurations. `check_command(card=..., variant=...)` selects the requested
card and weight set. Keep the image and hydrated Volume mounted in subsequent
jobs. The first exact design compiles the Triton draw kernel and captures CUDA
graphs; `MODEL_OPT_JIT_ROOT=/weights/jit` retains the compile cache.

## Qualify full-backbone design

Run the recipe's `CHECK` on a GPU after hydration. Upstream `check` supports
`exact` only; a stock comparison uses `design --mode off`. A successful dry
run establishes setup, while the design's startup probes establish that the
acceleration actually operates on this worker.

Start with the pinned clone's public two-monomer example:

```sh
cd /kit/proteinmpnn
bash run.sh check --config h100 --mode exact --variant vanilla
bash run.sh design --config h100 --mode exact --variant vanilla \
  --input "$MPNN_DIR/inputs/PDB_monomers/pdbs" --out /out/exact \
  --seed 37 --num_seq_per_target 1 --batch_size 1 --save_score 1 --save_probs 1
bash run.sh design --config h100 --mode off --variant vanilla \
  --input "$MPNN_DIR/inputs/PDB_monomers/pdbs" --out /out/off \
  --seed 37 --num_seq_per_target 1 --batch_size 1 --save_score 1 --save_probs 1
```

These counts are qualification inputs. For production, preserve the user's
counts, sequence temperatures, residue dictionaries, PSSM, constraints, and
other upstream options. The recipe's `design_command()` passes the supplied
options verbatim. Upstream designs whole batches, so the requested sequence
count should be a multiple of batch size. Include the number of temperatures
when determining the expected artifact count.

Require the requested variant and `[proteinmpnn-opt] ACTIVE mode=exact`,
successful startup probes, final `probe=PASS`, exit code 0, and
`opt_manifest.json`. Read the manifest's actual levers, `partial`, `opted_out`,
pins, and exit status. Never add `--allow-partial` or `--hybrid_gemm 0` merely
to produce a successful status. A changed optimization set needs an explicit
record of the user's selected tradeoff.

Count and parse `seqs/5L33.fa` and `seqs/6MRR.fa`. Each contains one native
reference and one designed record in this smoke. The native reference is not
an extra generated design. Parse the score and probability NPZ files with
`allow_pickle=False`. The [artifact validator](../scripts/validate_complexa_proteinmpnn.py)
checks those formats and emits output-relative file names and SHA-256 hashes:

```sh
python <installed-skill>/scripts/validate_complexa_proteinmpnn.py proteinmpnn \
  --out /out/exact --designs-per-target 1 --expected-targets 2
python <installed-skill>/scripts/validate_complexa_proteinmpnn.py proteinmpnn-compare \
  --off /out/off --exact /out/exact
```

The parity check compares FASTA bytes and every NPZ array. NPZ archive bytes
can differ because ZIP timestamps differ. Test `--variant soluble` with the
same qualification contract when that weight set is selected. This guide
does not imply that every variant, card, or scientific constraint has already
received paid qualification.

## Connect the result to the user's pipeline

Carry each backbone ID, PDB hash, designed chain identity, fixed-chain
identity, selected weight set, seed, and designed record index into the
prediction input. ProteinMPNN uses `/` between designed chains; resolve those
segments using the FASTA header's `designed_chains` and the probability
archive's `chain_order`. Fixed target chains stay linked to the source PDB.
Do not concatenate a target and binder merely because both are protein chains.
Parse one prediction input and its expected chain roles before scaling.

The [generated-complex handoff helper](../scripts/prepare_proteinmpnn_handoff.py)
prepares a binder-only design from an actual full-backbone generated complex.
It copies PDB bytes unchanged, or declares a CIF-to-PDB conversion with source
and converted hashes, every atom identity, actual author/label residue mapping,
and measured coordinate rounding. It refuses missing N/CA/C/O, numbering gaps
that would insert missing residues, ambiguous alternate atoms, or values that
cannot fit PDB fields. It never reconstructs atoms or samples another structure.
The controller is stdlib-only; its pinned native parser runs in the ProteinMPNN
environment, which supplies NumPy.

For a PXDesign output with independently observed target A0 and binder B0:

```sh
python <installed-skill>/scripts/prepare_proteinmpnn_handoff.py prepare \
  /work/generated-exact.cif /work/handoff --candidate pdl1-px-exact \
  --source-tool PXDesign --generator-mode exact \
  --binder-chains B0 --target-chains A0 \
  --chain-map '{"A0":"A","B0":"B"}' \
  --binder-placeholder-map '{"xpb":"UNK"}' \
  --native-parser "$MPNN_DIR/helper_scripts/parse_multiple_chains.py"
python <installed-skill>/scripts/prepare_proteinmpnn_handoff.py reference \
  /work/handoff /work/public-5o45.cif --target-chain A --reference-chain A \
  --start 1 --end 116 --numbering label
```

Chain roles and the reference crop come from that generator's actual request
and structure; adapt them to the user's input. The placeholder option applies
only when observed binder residues carry PXDesign's unknown `xpb` code. It
declares `xpb→UNK` in the PDB and the native parser's observed-unknown `-→X`
in the model input. All coordinates and the fixed target sequence remain
unchanged apart from declared PDB precision. Known amino-acid residues are
never replaced with an invented sequence. Original and normalized parser
files remain linked in `handoff.json`.

Mount the prepared directory at `/in/handoff` and use its parsed JSONL:

```sh
cd /kit/proteinmpnn
bash run.sh design --config h100 --mode exact --variant vanilla \
  --jsonl_path /in/handoff/parsed.jsonl \
  --chain_id_jsonl /in/handoff/chain_id.jsonl --out /out/exact \
  --seed 37 --num_seq_per_target 1 --batch_size 1 --save_score 1 --save_probs 1
python <installed-skill>/scripts/prepare_proteinmpnn_handoff.py verify \
  /in/handoff /out/exact
```

Stock ProteinMPNN replaces `chain_id_jsonl` assignments when `--pdb_path`
selects one PDB. Use `--jsonl_path` with the explicit designed/fixed-chain
dictionary to preserve those assignments. The helper's `commands.json`
provides the four bounded `vanilla`/`soluble` × `off`/`exact` qualification
commands. Preserve the user's counts, temperatures and constraints for their
scientific run. No provider is launched by the helper.

The verifier independently reads sampled `S`, design `mask`, and `chain_order`
from the actual probability NPZ without loading object arrays. It checks each
target residue against the parsed input and original reference crop, requires
target mask zero and binder mask one, and rechecks all input hashes. Combine
this with the FASTA/score/probability validator and Exact/Off parity above.

The [executed handoff evidence](evidence/accelerated-kits-2026-09-30/proteinmpnn-handoff.json)
covers real PXDesign Exact target A116/binder B75 and RFdiffusion3 Exact binder
A95/target B115 outputs. Both checkpoints and modes produced eight designed
sequences in total. All 924 fixed-target residue observations remained unchanged;
FASTA bytes and every NPZ array matched Exact/Off for both weight sets. Both
Exact passes engaged CUDA graphs and passed their numerical probes. This
qualifies the generated-backbone handoff; it does not establish affinity or
folding quality.

The kit supports vanilla and soluble full-backbone design in `off` and
`exact`. Its [documented capability boundary](https://github.com/anthropics/uplifting-biomolecular-modeling/blob/f4f62fa6592ae4938d49b1757bea0cfeff9f468e/proteinmpnn/README.md#modes)
is precise: `exact` refuses `--ca_only`, `--score_only`, probability-only
passes, tied positions, and positive backbone noise. C-alpha-only Genie3
outputs therefore require a selected C-alpha-compatible native route or an
explicitly requested reconstruction followed by separate validation.
Selecting `off` is another explicit option for unsupported upstream passes.
Do not silently reconstruct the backbone, select full-backbone weights, or
claim accelerated exact execution for that C-alpha lane. Preserve these
upstream capabilities for workflows that need them.

Record activation, parsed outputs, parity, downstream handoff, provider cost,
and cleanup as separate evidence. Keep account names, credential references,
local install paths, and raw provider records in the user's own run record.

Sources reviewed 2026-09-30: pinned upstream README, STOCK.md, Dockerfile,
requirements lock, run.sh, stock/PINS.json, and output writers.
