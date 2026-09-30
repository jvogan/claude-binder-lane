# Accelerated upstream builds

Upstream publishes faster builds of most tools this catalogue selects. They keep
the same weights and the same interfaces, and one of their modes is stated to
reproduce the unmodified model bit for bit, which the report's own check confirms
for 10 of the 11 configurations it tested. That matters here for two reasons. A
reproduction claim survives a speed-up that changes no output, for the
configurations where that identity was measured. And the release states pins and
digests for artifacts this catalogue carries as holes.

This page records the upstream release and the package's routes into it.
The small-campaign worker invokes the Boltz-2 and ESMFold2 kits. Modal recipes
also ship for Genie3, PXDesign, RFdiffusion3, BoltzGen, Complexa, and full-backbone
ProteinMPNN. Read the [integration guide](accelerated-kit-integration.md) to
prepare those kits and connect their artifacts to the scientist's workflow.
Other kits remain selectable through their documented native interfaces.

Read 2026-09-17.

## What the release is

`github.com/anthropics/uplifting-biomolecular-modeling`, licence Apache-2.0,
created 2026-09-17, one commit on `main`, `f4f62fa6`, "Initial public release of
the model-optimization kits". Thirty-six kit directories plus a shared `common/`
core. The repository states it is not maintained and does not accept pull
requests, so it is a snapshot at the upstream versions each kit pins.

Each kit directory holds `stock/` with the pinned upstream release unmodified,
`opt/` with the kit's own package, `environment/` with a `Dockerfile`, an
`apptainer.def` and a `requirements.lock`, `configs/` with one file per GPU card,
`STOCK.md` with the pins and the stack, and `CHANGES.md` with what each mode
changes.

Thirty-four kits engage inside the upstream Python environment, as the release
states. The two foundry kits, `rfdiffusion3` and `rosettafold3`, instead install a
file overlay into a second interpreter and keep a pristine interpreter for `off`.
Thirty-one kits ship a `<kit>_opt_autoload.pth` start-up hook, counted from the
tree at the pinned commit, and both foundry kits are among them. The five that
ship no such hook are `af2ig`, `ef2inv`, `esm_if1`, `progen2` and `proteinmpnn`.

Weights are not included. Every kit fetches them from the upstream provider under
that provider's own terms.

## The four modes

The release uses these mode names where the selected kit supports them.
Genie3 and RFdiffusion3 offer `off`, `exact`, and `fast`; ProteinMPNN offers
`off` and `exact`. Read the selected kit's mode and hardware contract before
dispatch. A reproduction claim requires its actual fidelity evidence.

| Mode | What it guarantees | Use it when |
| --- | --- | --- |
| `off` | Upstream exactly as published, no optimization engaged | Establishing a reference result |
| `exact` | Intended stock-output fidelity; the report measured bit identity for 10 of 11 configurations checked | A run needs a per-tool baseline comparison |
| `fast` | Numerics differ within the tool's own seed-to-seed spread | Cost matters more than bit reproducibility |
| `big` | Memory-oriented route; supported levers and output semantics depend on the kit | A supported input needs a smaller memory footprint |

`fast` is the default in most kits. Some kits expose multi-GPU `big` through
`--n_gpu P`; BoltzGen's kit refuses multi-device execution. For the report's
multi-GPU configurations, results match single-GPU runs within tolerance rather
than bit for bit. A `big` mode name alone establishes no multi-GPU capability.

**Read the `exact` guarantee as measured, not absolute.** The release states the
guarantee as outputs identical to `off`. The report measures it and reports a
weaker result: bit-identity of `exact` against `default` held on the
FoldBench-Lite inputs for 10 of the 11 configurations checked, with deterministic
settings enabled in both. The report also states that several of these models are
not deterministic at their standard inference settings, so `default` itself
differs run to run, and that the deterministic-settings check is how it separated
the two effects. Two consequences follow for this package. A reproduction claim
resting on `exact` needs the per-configuration result for the tool it names, not
the mode name alone. And the guarantee is measured against `default`, the tuned
baseline, rather than against `off`, so a run comparing `exact` to a plain
install is not the comparison the report made.

TODO: which configuration failed the 10-of-11 bit-identity check, and whether it
is a tool this catalogue selects. The report places the per-configuration result
in its supplementary information, which this page has not read. Reading the SI
section that Figure 3's bit-identity sentence cites would settle it.

A kit prints `[<kit>-opt] ACTIVE mode=<mode>` when a mode engages, or
`NOT ACTIVE: <reason>` and exits 3 when it cannot. `run.sh check --config <card>
--mode <mode>` resolves a mode without running the model. A weights file that
misses its pinned digest exits 1 naming the file. Those exit codes make a silent
fall back to stock detectable, which is what this package's smoke-before-scale
rule needs.

## Forward-pass speed-ups on H100 80 GB

Read from Figure 3 of the technical report. Every figure is a geometric mean over
the benchmarked input sizes, five for ESMFold2 and seven for the others, typically
200 to 1,400 tokens.

| Model | `exact` | `fast` | `big` |
| --- | --- | --- | --- |
| ESMFold2 | 1.7 | 4.4 | 3.9 |
| ESMFold2-Fast | 1.6 | 5.2 | 4.3 |
| OpenFold3-p2 | 1.7 | 5.1 | 5.1 |
| OpenBind-0 | 1.4 | 4.9 | 4.9 |
| AlphaFold3 (JAX) | 1.0 | 2.5 | 2.5 |
| AlphaFold3 (PyTorch) | 0.7 | 3.0 | 2.2 |
| Protenix v2 | 2.5 | 4.1 | 2.9 |
| Protenix v1 | 1.7 | 3.2 | 1.8 |
| OpenDDE | 1.7 | 2.3 | 1.6 |
| Boltz-2 | 1.8 | 5.6 | 4.1 |
| Chai-1 | 2.5 | 6.4 | 6.3 |
| RoseTTAFold3 | 1.9 | 3.4 | 1.5 |
| AtlasFold | 1.3 | 2.8 | 2.4 |
| ColabFold 1.6.1 | 1.1 | not shown | 4.2 |
| AF2 initial guess | 1.0 | 5.0 | 4.4 |

The two AlphaFold3 rows run with OpenFold3-p2 weights. The figure shows no
distinct `fast` bar for ColabFold 1.6.1. Pooled arithmetic means over models are
`exact` 1.6 over 14 models, `fast` 4.1 over 13, and `big` 3.4 over 14.

**`exact` is not free on every tool.** It is 1.0 on AlphaFold3 in JAX and on AF2
initial guess, and 0.7 on AlphaFold3 in PyTorch, which is slower than the
baseline. On OpenFold3-p2 it raises peak memory, because its bit-exact triangle
attention kernel trades memory for the guarantee. Read the per-tool figure before
selecting `exact` for a fidelity run.

**The baseline is a tuned one.** The report calls it `default`: the unmodified
model run in the fastest correct way available to a knowledgeable user on the same
GPU, with optional fused kernels enabled and a batch size that saturates the card.
It calls the model run exactly as its authors ship it `base`. A speed-up measured
here against `default` will look larger if compared against a plain install, so do
not read these ratios as what an untuned local route would gain.

The timed quantity differs by tool family. Structure prediction is timed as the
forward call per complex, structure generation as designs per GPU-second, and
inverse folding as the end-to-end wall time of a design pass including start-up.

## Pins and digests this release settles

Each kit carries `stock/PINS.json` with the upstream commit, artifact digests,
per-weights-file sha256 and byte counts, and the pinned stack. Those settle several
fields this catalogue records as `TODO` or `__REQUIRED__`.

`protenix_v2` pins `github.com/bytedance/Protenix` tag v2.0.0 at commit
`2475421477ab414b571149ad4a875c390ff8a35d`, wheel `protenix-2.0.0-py3-none-any.whl`
at sha256 `be4584dda688e0518ce12909d8d51d0a0868e7daa2bbb95cbac17440c0ac309f`, and
checkpoint `checkpoint/protenix-v2.pt` at 1,859,785,497 bytes, sha256
`8f931f9774a396b67033d0e58628e1834f4a1448165e04254b40a780b0c0d599`. Its stack is
Python 3.11.5, torch 2.13.0+cu130, triton 3.7.1, CUDA 13.0, cuequivariance 0.11.1.
Its recorded CLI defaults are `--seeds 101 --cycle 10 --step 200 --sample 5
--dtype bf16` with `LAYERNORM_TYPE=fast_layernorm`.

The same kit confirms independently that the upstream download server does not
serve that checkpoint, answering HTTP 403 while the data caches are served. Its
installer names the checkpoint absent and exits 1 until a holder of the file places
it, then digest-checks all seven files. So the digest above is a value to check a
file against, not a way to obtain one.

`rfdiffusion3` pins the `rfd3` model of `github.com/RosettaCommons/foundry` at
commit `4010e3e2e7350edada3e25a45c908c6bf407df4d`, thirteen commits after v0.2.0,
and checkpoint `rfd3_latest.ckpt` at 2,690,316,669 bytes, sha256
`9b3f85923e0d51e9453e15cdd2f8c666e7ce096a60577f57d11bbc54ae6d67c1`, served from
`files.ipd.uw.edu/pub/rfd3/rfd3_foundry_2025_12_01_remapped.ckpt`. It records that
upstream registers no checksum for that file and supplies its own.

`boltz2` pins `boltz` 2.2.1, `github.com/jwohlwend/boltz` tag v2.2.1 at commit
`cb04aeccdd480fd4db707f0bbafde538397fa2ac`, with `boltz2_conf.ckpt` at sha256
`090e82ac8c92f5e943fa1b39e7410a44027bea7243c0bbb3caa67a77fc1428e1` and
`boltz2_aff.ckpt` at
`dcc5cd3722b1c9eaa34267e4ae32f55cbbf1963f4c19319381ccfa30fdd2ca9e` under
`BOLTZ_CACHE`, and states that upstream releases code and weights under MIT.

`colabfold` pins `colabfold` 1.6.1 with `alphafold-colabfold` 2.3.13, and
AlphaFold2-Multimer v3 parameters `params/params_model_{1..5}_multimer_v3.npz`
from the 2022-12-06 release under CC BY 4.0, each with its own sha256. Its source
is `storage.googleapis.com/alphafold/alphafold_params_2022-12-06.tar`, which is
the same artifact this catalogue's `freebindcraft` entry already names at
5,587,968,000 bytes, while ColabFold's own fetcher pulls the
`alphafold_params_colab_2022-12-06.tar` variant. It records that a single-chain
query resolves to `alphafold2_ptm`, whose parameters the kit neither pins nor
fetches and reports as `NOT PINNED`.

**Read `PINS.json` for a digest, not the prose.** Several kits state in `STOCK.md`
that a wheel or source-archive sha256 is recorded in `PINS.json` where that file
carries none. `boltz2`, `rfdiffusion3` and `rosettafold3` all overstate this way.
Their weights digests are real; their code artifacts are pinned by version and
commit rather than by digest.

`esmfold2` pins Biohub `esm` 3.3.0 at commit `26b0bc2b` and a Biohub fork of
`transformers` 4.57.6 at commit `ef32577f`, with weights `biohub/ESMFold2`,
`biohub/ESMFold2-Fast` and `biohub/ESMC-6B` under MIT. Two facts here change a
bring-up decision. The `esm` pin is a development build carrying the
label 3.3.0 that is not published on PyPI. No kit file states either fact, so
both were checked directly on 2026-09-18. The pinned commit
`26b0bc2b771e3e419ea74f445a5f35cc094a1509` of `github.com/Biohub/esm` is dated
2026-07-28, and the PyPI release index for `esm` runs 3.2.3 then 3.4.0 with no
3.3.0 between them. And the `transformers` fork's own
repository is no longer publicly available, with the same commit served by
`huggingface/transformers`. This catalogue records no `transformers` pin for the
ESMFold2 route at all.

## What a Claude Science user can do with this today

**Boltz-2 kit on Modal.** Build the Boltz-2 kit as a Modal image
from the recipe this lane ships at `envs/boltz2_kit_gpu.py`, which is the kit's
own Dockerfile in Modal's builder form, pinned to the release commit. Modal runs
on your own account. Hydrate the weights once through the recipe's `HYDRATE`,
which calls the kit's own downloader and checks all four files against the
digests the release publishes. Then run the kit and **pipe what it prints
through `claude_binder.kit_engagement`**, which is the one step that separates an
accelerated run from a stock one:

    python3 -m claude_binder.kit_engagement assess \
      --mode fast --stdout run.out --exit-code "$rc" --require-engaged

It exits 3 and says why if the build you paid for is not the build that ran. Read
`envs/boltz2_kit_gpu.py` before the first build: the mode check needs a GPU and a
hydrated cache, the image is large because the shared core carries prebuilt
kernels, and the driver requirement is an open question recorded there.

Eight kit families have a Modal recipe: Boltz-2, [ESMFold2](esmfold2-kit-modal.md),
and the six [design and sequence kits](accelerated-kit-integration.md). These
recipes preserve the pinned stacks and available kit modes; a fixed Binder
graph binding is optional. The ESMFold2 recipe stages the three CUDA extension builds and
keeps the pinned weights on a Volume. Its source and H100 selection have offline
checks; the image build, hydration, GPU check, and prediction have not been run
through this recipe. Claude Science should qualify that route in the scientist's
own account before using it in a campaign. A successful ESMFold2 `DRY-RUN`
announces readiness only; use the captured prediction exit code and `ACTIVE`
line to assess a completed fold.

Three different things are easy to conflate here. Separating them keeps the
catalogue from claiming a capability it does not have, and keeps it from
reporting a blocker that does not exist.

**Already runnable, without these builds.** Claude Science ships platform skills
for several tools this catalogue selects, including `esmfold2`, `boltz`, `chai1`,
`openfold3`, `alphafold2`, `proteinmpnn` and `borzoi`. Claude Science can operate
those through the platform's native skill route. A catalogue `availability.status` of `unbound`
means this package has no adapter contract for the tool. It does not mean the
tool is unavailable. Likewise a tool with no route class in
[the route matrix](../claude_binder/route_matrix.py) has no shipped
profile binding, which is a statement about Binder dispatch rather than about
what a scientist can run.

Those platform skills wrap upstream tools, and none of them names one of these
kits. The ESMFold2 skill exposes a kernel backend choice, which the recorded
platform-skill inventory states.

TODO: whether the platform's configuration is the report's `default` baseline.
Settling it decides whether the ratios above are a gain still available on top of
what the platform already runs, or a gain the platform has partly taken. Nothing
read for this page states it. The report defines `default` as the unmodified model
run in the fastest correct way available to a knowledgeable user on the same GPU,
with optional fused kernels enabled and a batch size that saturates the card, and
calls the model run as its authors ship it `base`. The platform skills record no
batch-size or fused-kernel configuration against that definition, and a kernel
backend choice is not by itself evidence that the fused path is the default one
the skill selects. Reading each skill's recorded defaults against the report's
`default` definition would settle it. Until then, do not quote these ratios as a
platform-relative gain.

**Packaged small-campaign worker.** The
[fast path](small-campaign-fast-path.md) now ships a settings template and
`scripts/small_campaign.py`: it creates a roster from BindCraft2 ranked CIFs,
admits individual jobs under a shared spend ledger, invokes either accelerated
Boltz-2 or ESMFold2 kit with a named mode inside an admitted worker, scores
explicit CIF/PAE artifacts, and writes a control-gated rank and cost report.
It requires a provider controller to build the image, create each job and
capture actual billing receipts. Those kit worker paths have offline fake-kit
tests; neither recipe has a new paid end-to-end qualification from this work.
No full Binder graph profile selects a kit mode yet.

**What the selected kit requires.** A Linux x86-64 host with an NVIDIA GPU at
the driver floor the kit states, the weights fetched separately under the
provider's terms, and a locally built image, because no prebuilt image is
published. Modal can dispatch Binder work. RunPod can dispatch once a Binder
worker is qualified; Lambda needs a provider transport. Each can host a native
kit workflow directly. The procedure is the kit's own: build from
`environment/Dockerfile`, `bash run.sh install --weights DIR`, then `bash run.sh
check --config h100 --mode exact` before any real work.

**The remaining qualification.** For each selected kit, run a real input in
the scientist's account at a named mode and independently parse its declared
structure or sequence outputs. Capture the successful exit code, matching
activation and the kit's own manifest and counters. Carry one real output
through the selected next stage. Predictors supplying CIF and PAE can then use
the shipped scorers. Record provider identity, billable time and charge.
This is a paid GPU run and needs the plan-bound operator approval and remaining
campaign ceiling before dispatch. Offline source checks and fake-worker tests
do not establish a measured GPU route.

## Why the mode has to be checked rather than assumed

Accelerated and stock runs write the same output formats. Output presence and
exit status alone do not prove that requested acceleration engaged. Read the
kit's activation, manifest and runtime counters together. The kit prints
`NOT ACTIVE` in two completely different situations: when a
mode could not engage, and when `off` was requested and stock ran as asked. So
scanning for that string rejects legitimate stock runs, and trusting the exit
code accepts a run that asked for `fast` and served stock.

`claude_binder.kit_engagement` compares what was requested against what engaged
and returns one of ten states. Two are acceptable, `active` and
`stock-by-request`. Two are acceptable with a caveat.
`active-with-card-levers-off` is what an `exact` run on A100 produces, and it
means the published timings for that mode do not describe the run.
`active-partial` is a mode that engaged without every optimization, which the
kits accept only when the run passed `--allow-partial`, so this module accepts
it on the same condition and refuses it otherwise. Five are refusals:
`not-active`, `silent-stock`, `mode-mismatch` for a run that asked for one
accelerated mode and got another, `unattributable-exit` for a run whose exit
code removes the basis for the attribution, and `no-announcement` for a kit that
said nothing at all. It reads text, so it costs nothing and runs anywhere.

For a kit that prints per-lever lines, the checker also refuses an `ACTIVE`
summary contradicted by a failed lever or by a selected lever marked `off`.
An unselected lever may correctly be `off`, and a lever skipped for a documented
variant, route, or card reason does not alone invalidate the run. Exit 1 is a
general failed or incomplete command across kits; a weights digest mismatch is
one example.

The tenth state, `dry-run-ready`, reports a successful `check` command. For
example, ESMFold2 prints `[esmfold2-opt] DRY-RUN mode=fast ...` with no `ok`
word. This state means the requested mode passed the kit's readiness check;
no prediction engaged, so `--require-engaged` still refuses it. Capture the
process exit code and pass `--exit-code` for both checks and runs. A missing
code, any unsuccessful process exit (including an out-of-memory exit 137), or
an `EXIT` line that contradicts the process result cannot establish a
completed run, even if an `ACTIVE` line appeared earlier.

`unattributable-exit` is the one that catches a run which looks perfect. The kit
prints its `ACTIVE` line when the build starts, so that line is already on
stdout when a weights digest fails to match, when the command line is rejected,
or when the kernel census finds a stock fallback where a kit kernel was
required. Those exit 1, 2 and 5. Exit 3 beside an `ACTIVE` line is caught for
the same reason, because the kit is contradicting its own announcement unless a
partial signal explains it. Other nonzero exits are caught too. A reader who
stopped at the announcement would record a clean accelerated run whose weights
were not the published ones.

`mode-mismatch` and `active-partial` exist because the alternative was a
sentence that is false.
A run that asked for `exact` and got `fast` has to be refused, because `exact`
is the mode that reproduces the unmodified model. It still ran an accelerated
build, so reporting it as `not-active` would tell a reader the GPU minutes
bought nothing. A partial run is the same shape in reverse: the kit itself
accepts it under `--allow-partial` and names the optimization it left out, so
refusing it regardless would contradict the kit being read.

The kits do not share one punctuation for these lines. boltz2 writes
`NOT ACTIVE: <reason>`, colabfold writes
`NOT ACTIVE mode=off (stock: nothing applied)`, af3_torch writes
`NOT ACTIVE mode=<mode> reason=<reason>`, and one kit brackets the mode. The
parser treats the colon as optional and reads a `mode=` or `reason=` tail when
one is present, so a deliberate `--mode off` run is recognised as deliberate on
every kit rather than only on the one whose punctuation was read first.

## What this release does not settle

It ships no weights, so a checkpoint this catalogue cannot obtain stays
unobtainable. The Protenix v2 403 is unchanged.

It runs on Linux x86-64 with an NVIDIA driver at the floor each kit states, 525 to
580 depending on the CUDA stack, and every kit is configured and stated for the
H100 80 GB. Several kits require CUDA 13.0. Nothing here runs without an NVIDIA
GPU.

Prebuilt container images are planned and not published. Building locally is the
route today, and the ESMFold2 environment compiles three CUDA extensions from
source.

It publishes no cost figures. The report's speed-ups are per-call ratios on one
card, not campaign costs, and they exclude model load. This package's own measured
rates in [measured costs](measured-costs.md) remain the only cost evidence here.

## What this release settles for BindCraft2, and what it does not

BindCraft2 joined the catalogue on 2026-09-21 and no kit in this release wraps it.
The release is a snapshot at one commit dated 2026-09-17, four days before the
BindCraft2 checkout this catalogue read, and its thirty-six kit directories name
no BindCraft. So there is no `bindcraft2` kit to build and no mode to check. The
tool is not unaccelerated for that reason: it carries its own fused-attention
path at `bindcraft/af/accel.py`, with chunked attention and a fused kernel route,
and its `cuda12` and `cuda13` extras pull `cuequivariance-jax` with the matching
`cuequivariance-ops` wheel. That is the same fused-kernel library family several
kits use, engaged by the tool's own code rather than by a kit overlay.

**One kit still settles evidence BindCraft2 needs.** The `colabfold` kit pins
AlphaFold2-Multimer v3 parameters `params/params_model_{1..5}_multimer_v3.npz`
from the 2022-12-06 release, each with its own sha256, and names
`storage.googleapis.com/alphafold/alphafold_params_2022-12-06.tar` as the source.
BindCraft2's `bindcraft/model_weights.py` fetches that identical archive and loads
seven files from it: the same five multimer v3 parameter sets, plus
`params_model_1_ptm.npz` and `params_model_2_ptm.npz`, which it holds back for
validation and for refolding a binder without its target. So the kit's
`PINS.json` carries a checkable digest for five of the seven files a BindCraft2
campaign loads, from the same archive, and the two pTM files are outside what it
pins. The kit reports its own pTM parameters as `NOT PINNED` for the neighbouring
single-chain case, so neither file gains a digest from this release.

That is a digest to check a file against rather than a way to obtain one, the
same limit the Protenix v2 entry above records. It is still the cheapest visible
route to closing the `TODO(evidence)` on the BindCraft2 weights row, which asks
for the digest of each `params_*.npz` an installation holds.

TODO: whether the five sha256 values the `colabfold` kit publishes match the
files a BindCraft2 installation extracts. Nothing here has opened either. Reading
the kit's `PINS.json` against a hydrated `~/.cache/bindcraft/alphafold` settles
it, and a matching pair would let the BindCraft2 toolcheck cite a published
digest rather than only the digest it measured.

**`kit_engagement` still applies if a kit ever wraps it.**
`claude_binder.kit_engagement` parses the `[<kit>-opt]` announcement grammar
rather than one kit's punctuation, so it reads any kit id. Nothing in this
package routes BindCraft2 through a kit today, and no run has.

## The published design objective

The technical report states the objective its design runs were scored against.
Recording it here matters because this package already forces two of its terms.

The report's own figures use the ipSAE part alone: for each design, the mean over
three predictors of its best-of-five-seed ipSAE, taking the lower of the complex's
two directional values, with half the corresponding score against GDF-11
subtracted for the two GDF-8 targets. The three predictors are ESMFold2,
ESMFold2-Fast and Protenix v2, and they were fixed by the grader rather than chosen
by the designing model. Each selection is summarized by the median and the maximum
over its designs.

The benchmark's own score is broader: the mean over the 30 designs of a value in
which ipSAE is blended with DockQ against the submitted designed complex,
multiplied by a diversity factor, scoring zero for copies of known binders or of
the target's chains, and carrying a selectivity term for the two GDF-8 targets.

Every one of those terms already exists here. `ipsae_min` is the lower of the two
directional values and `sc_dockq` is the pose term, and a baseline-fidelity run is
required to rank on exactly those two, at a 4:1 ipSAEmin to sc-DockQ weight ratio.
The selectivity term exists too, as `selectivity_delta`, applied when a configured
target carries the `antitarget` role, with its own counter-screen weight rule. See
[scoring terms](scoring.md).

What is missing is not a field. No shipped profile template sets `counter_screen`,
so the selectivity path ships unexercised, and the same is true of the fidelity
path itself: all sixteen templates that carry `baseline_fidelity` set it false.

The report also states what a single run can and cannot show. Across five runs of
one model against one target, the 24-hour median differed by 0.046, taken as the
median over 48 model-target combinations, and by up to 0.304. Runs with the
accelerated builds scored between 0.012 and 0.028 higher in median than runs
without them. The run-to-run spread is larger than that difference, so one run
cannot attribute a score change to a build. The report adds that its two arms also
differed in reference sheets and code versions, so the difference is not
attributable to the accelerated builds alone.

## How to check this page

The repository state is one `gh api` call:

```
gh api repos/anthropics/uplifting-biomolecular-modeling \
  --jq '{license:.license.spdx_id, created:.created_at}'
gh api repos/anthropics/uplifting-biomolecular-modeling/git/trees/HEAD \
  --jq '.tree[] | select(.type=="tree") | .path'
```

The start-up hook count is one call over the same tree:

```
gh api "repos/anthropics/uplifting-biomolecular-modeling/git/trees/f4f62fa6?recursive=1" \
  --jq '[.tree[].path | select(endswith("_opt_autoload.pth"))] | length'
```

A kit's pins are one file:

```
curl -sL https://raw.githubusercontent.com/anthropics/uplifting-biomolecular-modeling/f4f62fa6/protenix_v2/stock/PINS.json
```

The speed-up table and the objective come from the technical report at
`www-cdn.anthropic.com/c03643714397d9d396fa1ce1794f5f9f7863a82c.pdf`, 20,188,137
bytes, md5 `eb5b6464d739d694a70e4adf56aec5fd`, 139 pages. Figure 3 is on page 5 and
its values are in the PDF text layer, so `pdftotext -layout -f 5 -l 5` reproduces
the table without reading the image. The objective is in section S8, and the
run-to-run spread is in S8.2.

An earlier revision of the same report is served at
`www-cdn.anthropic.com/d8ca26d0d205708d26c7337cf4cfe7cb52e9b671.pdf` at 20,187,474
bytes. Every number is identical between the two revisions. The later build adds
one sentence and one reference and renumbers the citations after it.
