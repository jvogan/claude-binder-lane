# Proteina-Complexa optimization kit on Modal

Claude Science can use the shipped [recipe](../envs/complexa_kit_gpu.py) for
standalone generation or connect it to the scientist's own design, scoring,
and prediction workflow. It preserves the [Anthropic kit](https://github.com/anthropics/uplifting-biomolecular-modeling/tree/f4f62fa6592ae4938d49b1757bea0cfeff9f468e/complexa)
at `f4f62fa6592ae4938d49b1757bea0cfeff9f468e`, Proteina-Complexa 1.1.0 at
`916eaaedce5b07c205efb6ef32370c01d366591e`, the bundled upstream archive,
ColabDesign, and the complete pinned dependency stack. A campaign adapter
can be added locally to the user's workflow; the direct kit is available
independently of that binding.

## Prepare the scientist's environment

Discover their connected Modal account, available helpers, approved spend
ceiling, and matching cached image or Volume. Resolve the installed recipe
path and build through the session's actual Modal helper:

```python
result = build_env(
    "complexa_kit_gpu",
    path="<installed-skill>/envs/complexa_kit_gpu.py",
    hydrate=True,
)
```

`build()` declares a Python 3.12.10 image, an account-local `/weights` Volume,
and runtime environment. Image assembly and hydration incur usage when
executed. The image keeps upstream at `LOCAL_CODE_PATH=/opt/pc`.
`CKPT_PATH=/weights/complexa` holds the two checkpoints, which total
7,034,391,160 bytes. `HYDRATE` stages those same public NVIDIA NGC sources
using curl with verified HTTPS, up to three fresh transfer attempts per file,
and exact size/SHA-256 checks before atomic publication. It then calls the
unchanged upstream kit installer for its own pin verification. This handles a
stock GnuTLS transfer failure that can otherwise leave a zero-byte checkpoint.
The NGC source redirects to `xfiles.ngc.nvidia.com`. Pass `volume_name` to `build()` to
use another account-local cache. Reuse the image and Volume only when their
pins match the requested stack.

The [Dockerfile](https://github.com/anthropics/uplifting-biomolecular-modeling/blob/f4f62fa6592ae4938d49b1757bea0cfeff9f468e/complexa/environment/Dockerfile)
uses torch 2.7.0+cu126 from the complete lock, including the PyG and JAX
direct-URL wheels and their digests. Preserve `PIP_CONSTRAINT` and its
hatchling 1.31.0 pin during source and editable installs. The recipe makes
Open Babel's `libxrender1`, `libxext6`, and `libx11-6` dependencies explicit.
Dropping these can create an image that installs successfully and fails when
the generation pipeline imports its data handling code.

Use H100 by default, or the upstream A100/H200 configurations with matching
hardware. Upstream's driver requirements are 560 or newer, or compatible
data-centre drivers from 525; inspect the worker's actual driver before a
design. `CHECK` requires hydrated weights and a GPU. It verifies pins,
checkpoint presence, and the autoload hook without loading a model. A design
then proves the mode's levers actually served sampling work. Keep release
and kit-specific licences with a redistributed image.

## Qualify default generation and AF2 reward on public input

After `HYDRATE`, run the recipe's `HYDRATE_AF2` in the built environment with
its `/weights` Volume attached. This stages the actual AF2 parameters required
by upstream's default reward. The optional dependency section below describes
that command and the complete parameter-set checks.

Upload the shipped [PD-L1 target entry](../examples/complexa-kit-pdl1.json) to
the worker as `/in/pdl1.json`. Its `/opt/pc/...` target path points inside the
pinned image and contains no user or account installation path. For another
image layout, write `target_path` using that image's actual `LOCAL_CODE_PATH`.
The entry uses the pinned upstream PD-L1 target, target crop A1-115, its four
documented hotspots, and an 80-residue binder.

```sh
cd /kit/complexa
bash run.sh check --config h100 --mode exact
bash run.sh design --config h100 --mode exact \
  --input /in/pdl1.json --out /out/exact -- \
  ++generation.dataloader.dataset.nres.nsamples=1 \
  ++generation.dataloader.batch_size=1 ++seed=5
python <installed-skill>/scripts/validate_complexa_proteinmpnn.py complexa \
  --out /out/exact --expected-designs 2 --binder-length 80 \
  --require-af2-reward
```

This qualifies **generation with the enabled default AF2 reward**. The bounded
sample and batch counts retain 400 sampler steps, best-of-n search with two
replicas, AF2-Multimer, three recycles, initial guess, and the default reward.
The recipe's `design_command()` adds no qualification settings. It passes every
supplied Hydra override unchanged. Preserve the user's seed, batch, sample
count, search, reward model, and constraints in production. `exact` and `big` retain the
upstream exact-output semantics; `fast` uses documented numerical changes.
The upstream default is `fast`, so name the scientific mode explicitly.

Use a fresh output directory for every run. Upstream can stop at an existing
results CSV or rewrite earlier PDBs. A successful process status against old
outputs proves no new generation. Require the requested `ACTIVE mode=...`,
the final `EXIT` with successful child and kit status, the expected number of
new designs, and `opt_manifest.json`. Keep its `gated` and per-lever counters
in the record; fixed binder length engages pair assembly, while padded length
ranges can leave that lever on its declared upstream path.

Parse the new PDB under
`inference/search_binder_local_pipeline_*/job_*/*.pdb`. Chain A is the target
crop and chain B is the binder. Require finite coordinates and N/CA/C/O for
each residue before treating the output as a full-backbone handoff:

The [validator](../scripts/validate_complexa_proteinmpnn.py) returns chain
roles, residue counts, output-relative filenames, and hashes. For exact
parity, run `off` with the same input, seed, batch, and overrides in another
fresh directory and compare corresponding PDB bytes and reward values. A two-replica qualification
establishes operability, not a production throughput or memory benchmark.
The stock manifest's `designs_expected=1` counts the requested sample and omits
the two-replica multiplier. Require two actual PDBs and two AF2 reward rows.

## Preserve full search and reward capabilities

The kit's weights step hydrates Complexa's denoiser and autoencoder. It does
not fetch the AF2 parameters required by upstream's default best-of-n reward
workflow. The recipe's optional `HYDRATE_AF2` runs the pinned upstream
downloader's `--af2` step with a scratch directory and symlink into
`/weights/af2`. It verifies archive integrity through upstream and checks all
15 expected parameter files. `AF2_DIR=/weights/af2` is set in the recipe's
environment, and the pinned checkout remains intact. Schedule this separate
hydration within the scientist's provider authorization and budget before
running the full reward workflow.

On another writable image or an existing upstream installation, the upstream
form is:

```sh
bash "$LOCAL_CODE_PATH/env/download_startup.sh" --af2
export AF2_DIR="$LOCAL_CODE_PATH/community_models/ckpts/AF2"
```

Keep AF2 parameters on a persistent store or a matching image, and set
`AF2_DIR` to the actual directory. The direct downloader's default path is
in the checkout; the recipe's `HYDRATE_AF2` redirects it to the Volume.
For optional evaluation/reward stages, configure their
real dependencies, such as `ESM_DIR`, `RF3_CKPT_PATH`, `RF3_EXEC_PATH`,
`FOLDSEEK_EXEC`, `MMSEQS_EXEC`, `DSSP_EXEC`, and `SC_EXEC`. The kit's
`/bin/true` placeholders satisfy eager configuration resolution during
generation; they provide no scientific reward or evaluation capability.
Check the enabled stage's files and executables before dispatch. Obtain the
shape-complementarity tool as upstream describes and point `SC_EXEC` to it.

Once dependencies are configured, retain the full selected search and reward
settings. Qualify the enabled scientific stages separately.

Require JAX's actual GPU device in the execution record and inspect the
resolved AF2 model/recycle configuration. The pinned composite reward model
catches an individual folding error and can continue with a zero total
reward. Check for its `Error computing reward from folding model` warning,
and reject that full-reward claim. Parse the saved rewards CSV for each
generated replica, require finite `af2folding_i_pae` and AF2 confidence
columns, and count the rows. `--require-af2-reward` enforces those artifact
checks without adding scientific quality thresholds.

The default AF2 reward uses the installed JAX and bundled ColabDesign with
real AF2 parameters. Five model parameter sets are available. The pinned
inference default passes model indices 0–4 to ColabDesign, whose unchanged
`num_models=1` and `sample_models=False` select model index 0 per reward.
The actual reward CSV records `af2folding_models_log=0` and
`af2folding_recycles_log=3`; availability of five models does not mean five
forward evaluations. Preserve a user's requested model selection and qualify
it explicitly when it differs. The default does not invoke optional DSSP/sc/MMseqs/Foldseek,
RF3, or ESM reward tools. Configure those when their stages are selected.
The kit's `design` verb invokes upstream `complexa generate`. Upstream's
separate `complexa design` command also runs filter, evaluate, and analyze;
those stages require their own enabled dependencies and output checks.

An optional **generation-only** workflow can be explicitly selected when the
scientific task calls for structures without AF2 reward:

```sh
cd /kit/complexa
bash run.sh design --config h100 --mode exact \
  --input /in/pdl1.json --out /out/generation-only -- \
  ++generation.search.algorithm=single-pass ++generation.reward_model=null \
  ++generation.dataloader.dataset.nres.nsamples=1 \
  ++generation.dataloader.batch_size=1 ++seed=5
python <installed-skill>/scripts/validate_complexa_proteinmpnn.py complexa \
  --out /out/generation-only --expected-designs 1 --binder-length 80
```

That selection qualifies one generated backbone and no reward computation.
It is a caller choice; the recipe never inserts it.

The [executed public-input evidence](evidence/accelerated-kits-2026-09-30/complexa.json)
records two generated and AF2-scored structures per `off`, `exact`, `fast`,
and `big` mode. Exact and Big PDB bytes and all non-path reward values matched
Off. Fast retains its numerical mode. All requested optimized levers served;
the Fast/Big pair-bias counters also retain 24 stock-guard calls for attention
without a wired bank or pair features. Separate filter/evaluate/analyze stages
and optional external tools were not executed in that qualification.

## Connect generation to the user's pipeline

Carry stable candidate IDs, PDB hashes, target crop and residue mapping,
hotspots, chain roles, and binder lengths into sequence design or prediction.
Complexa also decodes a sequence; the user may retain it, redesign chain B
with a selected full-backbone inverse-folding model, or compare both arms.
Keep chain A fixed when designing a binder unless the user requested another
scientific task. Parse one downstream input and output before scaling.

Record setup, requested mode, actual levers, generation, reward qualification,
downstream handoff, cost evidence, and cleanup separately. Keep provider
accounts, credentials, private paths, and raw provider records in the user's
own run record. A shared recipe carries reusable pins and commands only.

Sources reviewed 2026-09-30: pinned upstream README, STOCK.md, Dockerfile,
requirements lock, run.sh, stock/PINS.json, and output writers.
