# BoltzGen optimization kit on Modal

Claude Science can use the [Anthropic BoltzGen kit](https://github.com/anthropics/uplifting-biomolecular-modeling/tree/f4f62fa6592ae4938d49b1757bea0cfeff9f468e/boltzgen)
standalone, run its integrated pipeline, or adapt the scientist's own sequence
design and scoring workflow through a
[campaign-local connector](connector-authoring.md). The shipped
[recipe](../envs/boltzgen_kit_gpu.py) pins the kit to
`f4f62fa6592ae4938d49b1757bea0cfeff9f468e`.

## Preserve the complete stack

The [upstream Dockerfile](https://github.com/anthropics/uplifting-biomolecular-modeling/blob/f4f62fa6592ae4938d49b1757bea0cfeff9f468e/boltzgen/environment/Dockerfile)
uses standalone CPython 3.11.5, its complete requirements lock, the bundled
BoltzGen 0.3.2 wheel, torch 2.13.0+cu130, Triton 3.7.1, and the
cuEquivariance 0.11.1 CUDA 13 pair. A compiler remains available for Triton's
runtime launchers. The kit and shared core are installed editable from the
pinned tree.

BoltzGen's wheel metadata names CUDA 12 cuEquivariance distributions. The
kit's recorded stack intentionally uses the CUDA 13 pair, providing the same
modules against its CUDA 13 torch. Its
[stock pins](https://github.com/anthropics/uplifting-biomolecular-modeling/blob/f4f62fa6592ae4938d49b1757bea0cfeff9f468e/boltzgen/stock/PINS.json)
document the two-name `pip check` discrepancy. Preserve the CUDA 13 stack
and inspect actual imports, kernel evidence, and runtime versions.
Installing CUDA 12 packages to clear that metadata warning would change
the qualified stack.

H100 is the default recorded hardware. Upstream also configures A100 and
H200; choose the actual worker's corresponding `--config`. Upstream
requires a CUDA 13.0-capable host driver, version 580 or newer. Record the
observed driver. Persistent JIT caches reduce repeated kernel compilation,
while model processes still perform their own setup.

## Build, hydrate, and qualify

Discover connected compute and reusable suitable environments first. Resolve
the recipe path from the installed skill. A Claude Science Modal helper that
accepts user-supplied recipes can consume:

```python
result = build_env(
    "boltzgen_kit_gpu",
    path=installed_recipe_path,
    hydrate=True,
)
```

The recipe's `build()` also supports an authorized native Modal runner.
Keep the returned environment and weights Volume identity. `HYDRATE` calls
the kit's installer, filling approximately 8.4 GB of checkpoints and molecule
data in `/weights/boltzgen`. It verifies all six files against pinned Hugging
Face snapshots and full SHA-256 digests, then points cache refs at those
snapshots. Runtime variables are `BOLTZGEN_CACHE=/weights/boltzgen` and
`MODEL_OPT_JIT_ROOT=/weights/jit/boltzgen`. Card configs force offline
weight resolution. Include build and first hydration in the approved budget.

Run `HYDRATE`, then `check_command(card=..., mode=...)` on a GPU worker.
The `DRY-RUN` line proves readiness. For the smallest real `exact` or
`fast` qualification, use the bundled public PD-L1 spec:

```sh
cd /kit/boltzgen
bash run.sh design --config h100 --mode exact \
  /kit/boltzgen/opt/forward/fast_inference/tests/specs/pdl1_ref.yaml \
  --output /work/runs/boltzgen-exact \
  --num_designs 2 --seed 0 --diffusion_batch_size 2 --steps design
```

`smoke_command(output_dir=...)` constructs that command with shell quoting.
The kit requires a seed and a diffusion batch of at least two for `exact`
and `fast`. A diffusion batch of one refuses those modes. Upstream rounds
counts to whole diffusion batches, so this qualification requests two designs
explicitly. Sampling and recycling settings remain upstream's
defaults. Use a fresh output directory for unambiguous artifact evidence.

Require exit 0, the final `ACTIVE mode=... form=process` verdict,
`fallbacks=none`, parsed `opt_manifest.json`, and
`DESIGNS requested=2 produced=2 oom_skipped=0`. Keep `opt_configure.log`,
`opt_run.log`, and parse both generated structures. Exclude `*_native.cif`
input copies from the design count. Inspect the kernel census as well as
the launcher announcement. Upstream can skip out-of-memory batches and still
exit 0. Count complete outputs independently.

A generation-only run establishes the generator route. To qualify the
integrated inverse-folding, folding and analysis handoffs, remove
`--steps design` or call `smoke_command(output_dir=..., full_pipeline=True)`.
Verify each step's output and final sequence, complex and score artifacts
before scaling. For an `exact` equivalence test, compare seeded `off` and
`exact` runs with identical input specs and diffusion batches.

The shipped [artifact validator](../scripts/validate_rfdiffusion3_boltzgen.py)
checks actual structures, the design and kernel censuses, graph replay and
initialization counters, GPU step seeds, and CPU step completion.
`--full-pipeline` adds inverse-folded sequences, finite prediction arrays,
complex and binder-only refolds, matched analysis rows, and actual filtering
flags. `--check-smoke-defaults` checks the unchanged 500/200 sampling steps,
three recycles and filtering budget 30 for this qualification example.
Supply observed chain IDs and optionally the staged target PDB to verify
its preserved sequence. Successful receipts include relative names, hashes
and counts. They retain how many designs passed filters; ranked outputs
can include designs that failed those filters.

## Adapt the scientist's own workflow

`run.sh design` forwards upstream's spec, protocol, steps, checkpoints and
other `boltzgen run` settings. Pass per-step config overrides after a bare
`--`. Keep scientific settings explicit and record generated step configs.
The canned smoke input does not constrain the user to PD-L1 or one protocol.

`exact` preserves seeded stock execution; `fast` changes floating-point
execution for throughput; `big` lowers peak memory. Upstream documents
numerical changes for `big` on structure-conditioned specs. Multi-device
execution and in-process data loaders remain available through explicit
`off`. Report the exact reason when a kit mode is refused, resolve the
configuration with the scientist, and preserve requested-mode activation
verification when acceleration is requested.

For a custom downstream pipeline, identify the selected step's generated
complex, binder sequence and metric row. Generation alone need not supply
the final inverse-folded sequence. Keep engine-native mmCIF and configs.
If a consumer needs PDB, convert and independently verify atom counts,
coordinates, residue mapping and chain roles. Derive chain roles from the
input spec and output because chain letters vary across workflows.
Carry candidate IDs, stage parentage and hashes through sequence design,
folding and scoring. Freeze the target when redesigning only the binder.
Use the [candidate handoff contracts](connector-authoring.md) to bind the
result to the campaign executor, and prove that consumer on the smoke
artifacts before scaling.

Record input, weight and output hashes, kit and stock revisions, seed, mode,
hardware, image identity, elapsed time and measured provider cost. Offline
tests establish command construction. A GPU receipt establishes the modes
and outputs actually executed. Preserve upstream LICENSE, NOTICE and bundled
third-party notices in redistributed images.
