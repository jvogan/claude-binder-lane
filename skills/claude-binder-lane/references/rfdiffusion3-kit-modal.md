# RFdiffusion3 optimization kit on Modal

Claude Science can run the [Anthropic RFdiffusion3 kit](https://github.com/anthropics/uplifting-biomolecular-modeling/tree/f4f62fa6592ae4938d49b1757bea0cfeff9f468e/rfdiffusion3)
as a standalone generator, adapt an existing workflow, or create a
[campaign-local connector](connector-authoring.md). The shipped
[recipe](../envs/rfdiffusion3_kit_gpu.py) pins the kit to
`f4f62fa6592ae4938d49b1757bea0cfeff9f468e`. A missing Binder adapter describes
the packaged binding; it does not remove this native kit route.

## Preserve the complete stack

The [upstream Dockerfile](https://github.com/anthropics/uplifting-biomolecular-modeling/blob/f4f62fa6592ae4938d49b1757bea0cfeff9f468e/rfdiffusion3/environment/Dockerfile)
uses standalone CPython 3.12.1, its complete requirements lock, Foundry at
`4010e3e2e7350edada3e25a45c908c6bf407df4d`, torch 2.13.0+cu130,
Triton 3.7.1, and compiled NVIDIA apex. The image has two interpreters:
`/opt/stock/bin/python` contains pristine upstream RFdiffusion3;
`/opt/kit/bin/python` has the kit's ten-file overlay. `exact` and `fast`
use the overlaid interpreter. `off` uses the pristine interpreter named by
`RFDIFFUSION3_STOCK_PYTHON`. Installing plain Foundry in one environment
does not reproduce this kit.

The recipe compiles apex from its pinned source with CUDA 13.0 and records
the resulting wheel digest. That hash is specific to the build. Preserve
it as build evidence; do not claim the unpublished prebuilt wheel's hash.
`build_jobs=0` preserves upstream compiler scheduling; a caller can pass an
explicit build resource choice. Include image assembly in the approved
compute budget.

H100 is the default recorded stack. Upstream also ships A100 and H200 card
configs. Use the actual available card and corresponding `--config`.
Upstream specifies a CUDA 13.0-capable host driver, version 580 or newer.
Record the observed driver and torch/CUDA versions. A persistent
`MODEL_OPT_JIT_ROOT` keeps kernels between containers, while CUDA graph
capture and process startup occur again in each process.

## Build, hydrate, and qualify

Discover the scientist's connected compute and existing suitable environment
first. Resolve the recipe path from the installed skill. A Claude Science
Modal helper that accepts user-supplied recipes can consume:

```python
result = build_env(
    "rfdiffusion3_kit_gpu",
    path=installed_recipe_path,
    hydrate=True,
)
```

Use the returned environment identity and Volume for subsequent jobs.
The recipe's `build()` also supports an authorized native Modal runner.
It declares an account-local weights Volume and sets
`RFD3_CKPT=/weights/rfd3/rfd3_latest.ckpt`,
`RFDIFFUSION3_STOCK_PYTHON=/opt/stock/bin/python`, and
`MODEL_OPT_JIT_ROOT=/weights/jit/rfdiffusion3`. The kit's installer checks
the 2,690,316,669-byte checkpoint against its full upstream SHA-256.
An interrupted transfer can leave a short checkpoint that upstream keeps;
inspect and replace that failed file before retrying hydration.

Run `HYDRATE` and then `check_command(card=..., mode=...)` on a GPU worker.
A successful `DRY-RUN` proves readiness. Applied acceleration requires a
design run. `prepare_smoke_command(input_dir=...)` stages the public
upstream PD-L1 binder example, selects its PD-L1 entry, and resolves its
structure path. Then run:

```sh
cd /kit/rfdiffusion3
bash run.sh design --config h100 --mode exact \
  inputs=/work/smoke-inputs/pdl1.json out_dir=/work/runs/rfd3-exact \
  n_batches=1 diffusion_batch_size=1 seed=101
```

`smoke_command(output_dir=...)` builds this command with shell quoting.
It preserves upstream sampling settings and produces one real binder.
Use a fresh output directory: upstream skips existing designs, so a reused
directory can perform no model work and exit 5.

Require exit 0, matching `ACTIVE mode=...`, the ten-file overlay report,
`APPLIED` and `KERNELS` evidence, a complete `opt_manifest.json`, and
one parsed `.cif.gz` with its companion JSON. Count artifacts independently
of the exit code. Record input, checkpoint and artifact hashes, kit and
Foundry revisions, seed, hardware, image identity, elapsed time, and measured
provider cost. To test upstream's bitwise `exact` claim, compare
`off --det 1` and `exact --det 1` with the same inputs and seed.

The shipped [artifact validator](../scripts/validate_rfdiffusion3_boltzgen.py)
independently parses coordinates and full backbones, rehashes the files
against the kit manifest, and checks nonzero rollout, fused-transition,
initialization and CUDA-graph counters. The fast route also needs compile
and gather-kernel counters. Supply the actual binder and target chain IDs.
Its optional staged-target PDB check verifies the preserved target sequence.
Successful receipts contain relative artifact names, hashes and counts.

## Adapt the scientist's own workflow

Pass RFdiffusion3's full `key=value` settings through `run.sh design`. The
smoke helper is a bounded qualification example; it imposes no production
candidate, length, or inference setting limit. Keep the selected mode explicit.
`exact` preserves the deterministic recipe. `fast` changes floating-point
execution for throughput. Record the choice in provenance.

Upstream refuses kit modes for classifier-free guidance, low-memory mode,
the symmetry sampler, and upstream `compile_model=true`. Those requests
remain available through explicit `off` execution. Report the refused
combination and resolve the scientific configuration with the scientist.
Preserve their input settings and budget. Do not silently enable
`--allow-partial`, disable apex, or substitute another mode.
Exit 3 or 5 requires inspection of the stated reason.

For sequence design, retain engine-native mmCIF and JSON, and bridge the
coordinates to the downstream consumer's format. The shipped
`claude_binder.adapters.rfdiffusion3_generator` contains an mmCIF-to-PDB
converter. Use it through a connector with checks for the generated structure
and chain mapping. Before full-backbone ProteinMPNN, parse the PDB, verify
N/CA/C/O atoms, identify target and binder from the spec and coordinates,
and freeze the target sequence. Carry candidate IDs, parent IDs, residue
mapping, chain roles, source hashes and bridge hashes. Prove the handoff
on the qualified binder before scaling. A generated binder does not
establish folding or binding quality.

Offline recipe tests establish command construction and stack preservation.
A GPU receipt establishes the commands, modes and artifacts actually
executed. Preserve upstream LICENSE, NOTICE and bundled third-party notices
when redistributing an image or derivative.
