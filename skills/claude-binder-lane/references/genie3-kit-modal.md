# Genie3 acceleration on a user's compute

Claude Science can run this kit as a standalone generator, a stage in an existing
pipeline, or a campaign-local connector. The supplied Modal recipe preserves the
Anthropic release's generation stack. The user's compute account, approved cost
ceiling, target, mode and output destination are session inputs.

The recipe and input example are implemented here. A successful image build or
`check` is setup evidence. A real design requires activation, parsed structures,
hashes and a receipt. Read any dated execution evidence before claiming that a
particular card, mode or application integration has been qualified.

## Stack and setup

Use [the Modal recipe](../envs/genie3_kit_gpu.py). It pins the kit to
`f4f62fa6592ae4938d49b1757bea0cfeff9f468e` and stock Genie3 to
`d77ae5ac04212ff1e8b29b585859a3244c614804`. It reproduces CUDA 12.6.3 with
cuDNN, CPython 3.10.13, torch 2.7.1+cu126, Triton 3.3.1, Lightning 2.6.5
and the complete upstream lock. The kit configures H100, A100 and H200. H100
80 GB is the upstream reference card. Match `--config` to the selected card.
The host must support the upstream CUDA stack.

Resolve the user's connected Modal account through platform discovery. Import
the recipe and call `build()` to obtain its image, Volume mounts and runtime
environment. These declarations do not start a job. Image construction, weight
hydration, the GPU dry run and inference consume provider resources when run.
Use the authorized controller's resource, timeout, retry, duplicate and cost
bounds for every launch.

Run `HYDRATE` with the `/weights` Volume attached before `CHECK`. Hydration uses
the kit's own downloader and checks the checkpoint and config against upstream
SHA-256 pins. `GENIE3_WEIGHTS=/weights/genie3/pretrained/v1` names the directory
containing `config.yaml` and `checkpoints/step=600000.ckpt`. The weights are
about 0.4 GB. The runtime downloads them from Hugging Face; build-only hosts
are separate from the recipe's job egress declaration.

Use `bash -c` for job commands. A login shell can overwrite the image's PATH
and bypass `/opt/venv_g3`. Keep `GENIE3_ROOT=/opt/genie3` and the recipe's
environment. `MODEL_OPT_JIT_ROOT=/weights/jit` retains compilation caches between
containers. `fast` compiles for encountered complex lengths and batch sizes;
`exact` and `off` do not compile. The upstream image installs generation.
If a user's request includes upstream sequence prediction, sidechain prediction
or evaluation, prepare the corresponding upstream packages and databases in a
derived environment before selecting that route.

## Modes and a real qualification

| Mode | Upstream behavior |
| --- | --- |
| `off` | Clean stock `genie3 generate` subprocess with stock proof |
| `exact` | Bit-identical PDBs to stock at the same seed and batch size |
| `fast` | Further acceleration with documented numeric differences |

Preserve the mode requested by the scientist. The upstream default is `fast`.
Passing `exact` explicitly is useful when verifying equality against `off`.
Do not infer scientific equivalence or a speedup from a one-design smoke.

Mount [the public PD-L1 example](../examples/genie3-kit-pdl1.yaml) as
`/inputs/genie3-kit-pdl1.yaml` and a durable output directory at `/outputs`.
The request uses the pinned checkout's public problem, one sample and the full
100-step sampler. Run with the image's Python environment intact:

```bash
cd /kit/genie3
bash run.sh check --config h100 --mode exact
bash run.sh design --config h100 --mode exact \
  --input /inputs/genie3-kit-pdl1.yaml \
  --out_dir /outputs/genie3-exact --seed 101 --det 1
```

Native Docker or Apptainer use remains available through the pinned
[upstream setup instructions](https://github.com/anthropics/uplifting-biomolecular-modeling/blob/f4f62fa6592ae4938d49b1757bea0cfeff9f468e/genie3/README.md).
An existing compatible environment can use `genie3-opt design` or
`GENIE3_OPT=<mode> genie3 generate` with upstream arguments. Verify the shared
core and stock pins through `run.sh install` before use. The Modal recipe is
one setup route; it does not prescribe a pipeline graph.

## Qualification and downstream handoff

Require exit zero and `[genie3-opt] ACTIVE mode=exact`, plus
`opt_manifest.json` recording the same mode, completed status and requested
output count. For the example, parse the one PDB under
`/outputs/genie3-exact/04_pdl1/pdbs/`. Record target and generated chains,
residue counts, finite coordinates, the candidate ID, composed `request.yaml`,
weight and input hashes, hardware, stack, timings, logs and provider cost evidence.
Check all mode levers and refusals in the manifest and logs. `check`'s
`would_activate=True` describes a dry run, not inference activation.

For an equality qualification, use a fresh output directory for `off`, the
same request, seed, determinism and batch size, then compare the PDB bytes.
For throughput measurement, repeat `fast` after compilation on the same
lengths and use the batch size authorized for the scientific run. Preserve
the full sampler and report warm and cold timings separately.

The generated binder in this request is **C-alpha-only**. The pinned upstream
target featurizer creates one CA token and a three-coordinate position per
generated residue. The PDB writer preserves those atom types; it does not
reconstruct N, C or O atoms. Target tokens can carry a different atom
representation, so inspect the binder chain separately when checking backbone
coverage. An atom-bearing target does not establish a full binder backbone.
This contract is visible in the pinned
[feature construction and PDB writer](https://github.com/aqlaboratory/genie3/blob/d77ae5ac04212ff1e8b29b585859a3244c614804/src/genie3/generation/utils/feat_utils.py).

For sequence design directly from this output, prepare a compatible **CA-only
ProteinMPNN checkpoint** and its CA-only input parser. Use upstream's native
CA route or the ProteinMPNN kit's explicit `--mode off --ca_only` route with
the CA weights resolved. Record that sequence stage as a native or stock
CA-model substitution. The separate full-backbone ProteinMPNN `exact` kit
refuses `--ca_only` and does not accept this binder directly. Its
[mode contract](https://github.com/anthropics/uplifting-biomolecular-modeling/blob/f4f62fa6592ae4938d49b1757bea0cfeff9f468e/proteinmpnn/README.md)
and [named refusal](https://github.com/anthropics/uplifting-biomolecular-modeling/blob/f4f62fa6592ae4938d49b1757bea0cfeff9f468e/proteinmpnn/opt/proteinmpnn_opt/settings.py)
preserve that boundary. Confirm the CA checkpoint is present; the
full-backbone environment's weights do not establish CA-model availability.

If the user's chosen sequence model requires N, CA, C and O, prepare a
scientifically appropriate backbone reconstruction as a distinct stage,
qualify its complete atom coverage and geometry, and record its provenance
before passing it to that model. A file-format conversion cannot supply the
missing atoms. Keep target residues fixed, expose the requested binder
positions and retain chain/residue mappings. Downstream cofolding or scoring
needs the actual designed sequence and the same target identity. Validate the
consumer's parsed outputs at one candidate before scaling.

Kit modes refuse beam search, sidechain prediction and `--num-devices > 1`
with exit 3. The upstream `off` route supports upstream computations when their
dependencies are installed. For accelerated multi-GPU generation use upstream
dataset shards, one process per GPU with recorded seeds. A capacity refusal
names the batch that fits. Explain the limit and obtain the scientist's choice
before changing a requested mode or batch. An absent fixed Binder adapter is
an integration task; use the native route or
[campaign-local connector guidance](connector-authoring.md) to join the user's
chosen workflow.

Pins, mode contracts and command syntax come from the pinned
[Dockerfile](https://github.com/anthropics/uplifting-biomolecular-modeling/blob/f4f62fa6592ae4938d49b1757bea0cfeff9f468e/genie3/environment/Dockerfile),
[stock record](https://github.com/anthropics/uplifting-biomolecular-modeling/blob/f4f62fa6592ae4938d49b1757bea0cfeff9f468e/genie3/STOCK.md)
and [launcher](https://github.com/anthropics/uplifting-biomolecular-modeling/blob/f4f62fa6592ae4938d49b1757bea0cfeff9f468e/genie3/run.sh).
