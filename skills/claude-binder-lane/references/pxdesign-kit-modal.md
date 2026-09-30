# PXDesign acceleration on a user's compute

Claude Science can install this kit on the user's connected compute and connect
it to the user's chosen design, sequence and scoring workflow. The supplied
Modal recipe preserves the published Anthropic generation stack and every
supported mode. Resolve account identity, authorization, target and requested
outputs through the current session.

The recipe and public input are implemented here. Verify dated execution
evidence separately for each card and mode. An installed recipe, successful
image build or dry run does not establish usable inference outputs or a
Claude Science application integration.

## Stack and setup

Use [the Modal recipe](../envs/pxdesign_kit_gpu.py). It pins the kit to
`f4f62fa6592ae4938d49b1757bea0cfeff9f468e`, PXDesign to
`f788441313c84c3074fe9596ac2433f96b15c763`, Protenix to
`d18aa1daadd02a001b32bc7fa2278fc8a2f8f025` and PXDesignBench to
`f6d0d72496c23ca1374df5943754f3454ae8c549`. The kit driver has the explicit
stock-isolation repair described below; record the base kit commit and local
patch digest together. The exact upstream stack uses
CUDA 12.1.1 development tools, CPython 3.11.5, torch 2.3.1+cu121,
Triton 2.3.1, NumPy 1.26.3 and DeepSpeed 0.15.4. H100 80 GB is the upstream
reference card; A100 and H200 have upstream configurations. Use the matching
card configuration and a host driver supporting the CUDA 12.1 stack.

Call `build()` for the image, `/weights` Volume mount and runtime environment.
The declarations launch no job. Keep launches within the authorized user's
provider, resource, retry, time and cost limits. The development base is
necessary: stock Protenix's fast LayerNorm CUDA extension is compiled at image
build time using upstream's own import and architecture list. Keep this
extension and the upstream `--use_fast_ln` setting; changing it would alter
the pinned stock computation. PXDesignBench keeps its git metadata because
the stock pin check reads it. Upstream web UI fonts are removed in the same
installation layer, as in the upstream Dockerfile.

Attach the Volume and run `HYDRATE`. The kit fetches and SHA-256 checks four
checkpoints and three CCD-cache files, about 3.7 GB in total. Keep
`PXDESIGN_CKPT_DIR=/weights/pxdesign/checkpoint` and
`PROTENIX_DATA_ROOT_DIR=/weights/pxdesign/ccd_cache`. Downloads use
`pxdesign.tos-cn-beijing.volces.com`. `CHECK` needs a GPU even though it loads
no model. Use `bash -c` to preserve the image PATH. The image includes the
compiled LayerNorm, and the kit itself compiles nothing at inference time.
`MODEL_OPT_JIT_ROOT=/weights/jit` and `/weights/triton` retain runtime caches.

## Modes and a real qualification

| Mode | Upstream behavior |
| --- | --- |
| `off` | Stock PXDesign caller in a separate subprocess with stock proof |
| `exact` | Stock-identical computation under `--det 1` |
| `fast` | Further acceleration with documented numeric differences |
| `big` | `fast` numerics with pair-plane memory evaluated in slabs |

Preserve the mode requested by the scientist. The upstream default is `fast`.
A user can select `big` for large inputs; an out-of-memory failure never changes
the mode automatically. The kit forwards upstream inference options, including
sample count, diffusion steps, dtype and diffusion chunk size.

Mount [the one-task public PD-L1 input](../examples/pxdesign-kit-pdl1.json) at
`/inputs/pxdesign-kit-pdl1.json` and a durable output directory at `/outputs`.
It is the first task of the pinned kit's public `tasks_3targets.json`: PD-L1
chain A, residues 1–116, binder length 75 and the same upstream hotspots.
The target file is shipped inside the kit. Use the kit directory as the working
directory so the target's relative path resolves. The command keeps the full
400-step sampler and upstream bf16 dtype while qualifying one output:

```bash
cd /kit/pxdesign
bash run.sh check --config h100 --mode exact
bash run.sh design --config h100 --mode exact \
  -i /inputs/pxdesign-kit-pdl1.json -o /outputs/pxdesign-exact \
  --seeds 101 --N_sample 1 --N_step 400 --dtype bf16 --det 1
```

Every execution needs a fresh output directory. Upstream treats an existing
completed output as a resumed task and can exit zero without producing a new
design. Track task, seed and sample IDs when resuming a real campaign.

The pinned [upstream setup](https://github.com/anthropics/uplifting-biomolecular-modeling/blob/f4f62fa6592ae4938d49b1757bea0cfeff9f468e/pxdesign/README.md)
also supports Docker, Apptainer and a compatible host environment. Native calls
may use `pxdesign-opt design`, or `PXDESIGN_OPT=<mode> pxdesign infer` after
sourcing the matching card config. A native route can integrate with any
appropriate pipeline; it does not require the packaged Binder graph.

## Qualification and downstream handoff

Require exit zero, `[pxdesign-opt] ACTIVE mode=exact` and a matching
`opt_manifest.json` with activation and requested mode recorded. The one-task
example must report `DONE designs=1 ... expected=1` and yield one parsed mmCIF
under `/outputs/pxdesign-exact/pdl1/seed_101/predictions/`. Confirm that `ERR/`
contains no failed task. `DRY-RUN` and `PINS pinned=1` describe setup checks;
they do not establish that the kit ran.

Record finite atom coordinates, target and generated chain identities, residue
counts and backbone atoms, then hash the input, structures, manifest and logs.
Include the target crop and hotspot numbering, candidate IDs, seed, full argv,
stack, hardware, timings and provider cost evidence in the receipt. Activation
must record all required levers or a named refusal. Exit 3 and `NOT ACTIVE`
need a specific repair; they must never become an unannounced stock execution.

An equality comparison uses a fresh `off` output directory with the same input,
seed, `--det 1`, steps, sample count, dtype and LayerNorm option. Compare the
actual tensors or coordinate fields that the equality claim covers. mmCIF
headers can include run metadata; inspect them before treating file byte
differences as numerical differences. Report a scientific equivalence or
throughput claim only with its own measurements.

Check the full `stock_env_proof.json` before declaring stock isolation. The
unmodified kit at this pin has an unused `from . import stack` inside
`infer_loop.run`. The deterministic stock route therefore imports
`pxdesign_opt.stack`, which its after-call proof forbids. The proof reports
`after_ok=false` even with a clean initial environment, no carried lever
module loaded and exit zero. Initial qualification preserved that failure
alongside exact/off artifact equality. Identical coordinates do not repair an
isolation proof. The defect follows the pinned
[stock caller](https://github.com/anthropics/uplifting-biomolecular-modeling/blob/f4f62fa6592ae4938d49b1757bea0cfeff9f468e/pxdesign/opt/pxdesign_opt/stock_infer.py)
and [deterministic loop](https://github.com/anthropics/uplifting-biomolecular-modeling/blob/f4f62fa6592ae4938d49b1757bea0cfeff9f468e/pxdesign/opt/pxdesign_opt/infer_loop.py).

The supplied recipe removes only that unused import after installing the
pinned kit. Its [audited diff](../envs/patches/pxdesign-stock-isolation-unused-import.patch)
has SHA-256 `39c8e5f7bd83ceb735c0771bb041fd9ea68e56766185acab3547a75b17885a68`.
Build assertions check the complete file before and after, verify that the
function never reads `stack`, and refuse source drift. All other driver syntax,
determinism actions and activation APIs remain identical. No model source,
dependency pin, seed, step count or numeric policy changes. `META.local_patches`
and `/kit/pxdesign/local-patches.json` record the base commit, diff digest and
before/after file digests. Keep this provenance in the run receipt; the kit
driver is locally repaired rather than pristine upstream.

A native installation made from the upstream Dockerfile needs the same
explicit repair to pass this isolation check. Mount the audited diff at the
path below, verify its digest and both complete source digests, and apply it
from the directory containing the `pxdesign/` kit tree:

```bash
set -e
cd /kit
echo '39c8e5f7bd83ceb735c0771bb041fd9ea68e56766185acab3547a75b17885a68  /inputs/pxdesign-stock-isolation-unused-import.patch' | sha256sum -c -
echo '6b7f5b5cf251c38e452b2ab063f544f43d79b48684414d726b0f46e2743f638d  pxdesign/opt/pxdesign_opt/infer_loop.py' | sha256sum -c -
patch -p1 < /inputs/pxdesign-stock-isolation-unused-import.patch
echo '4e489b454adcc33cd67f497253107042a572a47c28e120563f7c26a6512fbe32  pxdesign/opt/pxdesign_opt/infer_loop.py' | sha256sum -c -
```

Qualify all four modes again with fresh outputs, `--det 1`, the full 400-step
sampler and the same source fixture. Require the full stock proof's initial
`ok` and `clean`, `after_ok=true`, empty after-call kit/lever module lists and
exit zero. Preserve acceleration probes and exact/off artifact comparison as
separate evidence. An offline import regression establishes the repair's
scope; actual model execution establishes its qualification.

On 2026-09-30, the repaired recipe passed all four modes on H100 with fresh
outputs, seed 101, `--det 1`, `--N_sample 1`, full `--N_step 400` and bf16.
The stock proof passed its initial and after-call isolation checks, with no
kit or lever module in the after-call census. Actual driver hashing confirmed
the repaired source, and every mode's structure was identical in bytes and
coordinates to its corresponding unmodified-driver output. Exact and off
also matched each other. Each structure contained 1,230 finite atoms with
full backbone coverage for target A0, 116 residues, and binder B0, 75 residues.
The original failed isolation proof remains separate historical evidence.
See the [dated qualification](accelerated-kit-qualification-2026-09-30.md).
This one-output check establishes execution and repair scope; repeated warm
throughput, maximum-size inputs and biological quality need their own evidence.

Preserve mmCIF metadata for downstream consumers. If a sequence designer needs
PDB, perform and record a validated conversion with a residue/chain mapping.
Verify that all requested backbone positions are represented, keep the target
fixed and design the user's selected binder positions. Cofolding and scoring
use the designed sequence, the original target identity and the recorded chain
roles. Qualify each consumer with one candidate before scaling. A missing
packaged adapter calls for the native tool or a
[campaign-local connector](connector-authoring.md), prepared by Claude Science
for that user's input and output contracts.

Pins and setup details come from the pinned
[Dockerfile](https://github.com/anthropics/uplifting-biomolecular-modeling/blob/f4f62fa6592ae4938d49b1757bea0cfeff9f468e/pxdesign/environment/Dockerfile),
[stock record](https://github.com/anthropics/uplifting-biomolecular-modeling/blob/f4f62fa6592ae4938d49b1757bea0cfeff9f468e/pxdesign/STOCK.md)
and [public task fixture](https://github.com/anthropics/uplifting-biomolecular-modeling/blob/f4f62fa6592ae4938d49b1757bea0cfeff9f468e/pxdesign/opt/forward/hoist/inputs/tasks_3targets.json).
