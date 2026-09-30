# Tool catalogue details

This page contains the operator bring-up and per-tool integration records moved
from [Tool catalogue](tool-catalogue.md). The main page keeps the current route
matrix, package status, profile, adapter, and stack-selection view.

## Main catalogue sections

The current sections remain on [Tool catalogue](tool-catalogue.md). These short
entries keep links in the moved records navigable.

## Derive the route matrix

[Open the current route matrix](tool-catalogue.md#derive-the-route-matrix).

## Route shapes

[Open the current route shapes](tool-catalogue.md#route-shapes).

## What Claude Science already ships

[Open the current Claude Science inventory](tool-catalogue.md#what-claude-science-already-ships).

### The other shipped skills

[Open the current shipped-skills list](tool-catalogue.md#the-other-shipped-skills).

## NVIDIA BioNeMo NIM

[Open the current NVIDIA BioNeMo NIM route](tool-catalogue.md#nvidia-bionemo-nim).

## What the package binds

[Open the current package bindings](tool-catalogue.md#what-the-package-binds).

## Which profile to start from

[Open the current profile guide](tool-catalogue.md#which-profile-to-start-from).

## Which adapters are wired

[Open the current adapter guide](tool-catalogue.md#which-adapters-are-wired).

## Tools you bring up yourself

These tools support the published protocol but do not ship as pre-installed platform skills. Plan for initial bring-up time and follow [tool bring-up](tool-bringup.md) to establish PASS status.

| Tool | Role | Licence | Note |
| --- | --- | --- | --- |
| RFdiffusion3 | Backbone generation | BSD-3-Clause code, weights licence unstated | The checkpoint host states no weight licence. Do not inherit older RFdiffusion terms |
| Genie3 | Co-design generation | Apache-2.0 code, Apache-2.0 weights declared in the model card | Emits a C-alpha trace in binder mode. See the [wiring contract](#genie3-wiring-contract) |
| Protenix v2 | Co-folding | Apache-2.0 for code and parameters | The first call compiles a CUDA kernel for 4 to 6 minutes on an H100 |
| ipSAE | Interface scoring | MIT | Single script with numpy dependency. Runs on CPU |
| DockQ | Pose and interface scoring | MIT | Pin v2. Versions v1 and v2 report different numbers |
| MMseqs2 | Novelty search | MIT | Backs the novelty gate. Record the database release with the verdict |
| BindCraft, FreeBindCraft | AF2-backprop binder design | MIT code with non-commercial component chain | Filters require PyRosetta or `--stage-all` |
| BindCraft2 | AF2-backprop binder design, structure and sequence together | Source-available and hosting-restricted code, MIT and CC BY 4.0 weights | A separate tool from BindCraft and from FreeBindCraft, with its own licence and no PyRosetta anywhere. Claude Science installs it on your compute route and drives that installation. See [BindCraft2 bound on 2026-09-21](#bindcraft2-bound-on-2026-09-21) |

### Two hosted generators bound on 2026-09-12

Both served for the first time on 2026-08-27 and 2026-08-28 through hosted profiles that close at `contract-test` claim level, and neither of those profiles declared production scoring. Both now have a catalogue entry and an adapter module this package ships, bound in `rfdiffusion3-two-arm.template.json`. Neither has run under this package's contract.

| Tool | Role | Pins | Claim level |
| --- | --- | --- | --- |
| PXDesign | Backbone generation | Source `f788441313c84c3074fe9596ac2433f96b15c763`, Protenix `d18aa1daadd02a001b32bc7fa2278fc8a2f8f025`, PXDesignBench `f6d0d72496c23ca1374df5943754f3454ae8c549`, DeepSpeed 0.15.1, Python 3.11 | `contract-test` |
| Proteina-Complexa | Sequence and structure co-design | Source `54058860d43444c7289873f77d3e50b5b02348cd`, JAX 0.4.29, JAXlib 0.4.29+cuda12.cudnn91 | `contract-test` |

PXDesign returns a backbone contract without sequences. Its raw mmCIF contains the 115-residue target and a 64-residue binder chain of unsequenced placeholders, requiring downstream sequence design before sequence-based scoring. Preserve the raw mmCIF, publish a backbone manifest, and avoid serializing the target chain as the candidate FASTA. Apply ProteinMPNN canonical residue names to the redesigned chain before comparing against reference poses.

The dated external Proteina-Complexa route also served `supplied-single-chain-v1`: one normalized contiguous protein chain A numbered from 1, explicit hotspots, a fixed 64-residue binder, and an explicit seed. `claude_binder.adapters.proteina_complexa_generator` now serves that contract, and it refuses a target chain other than A, a binder length other than 64, and any task name the application did not answer. Its code is Apache-2.0, while its weights (`complexa.ckpt` and `complexa_ae.ckpt`) fall under the NVIDIA Open Model License. Verify those terms before using weights outside private evaluations.

PXDesign exposes no requested generation seed, so its receipts record `seed_delivered: false` and `used_seed: null`. The external Proteina-Complexa contract accepts and reports a seed. ProteinMPNN and Chai-1 also record seeds.

### FreeBindCraft bound on 2026-09-12

FreeBindCraft is the BindCraft fork at `cytokineking/FreeBindCraft` that runs the BindCraft design loop without PyRosetta: relaxation through OpenMM, packing through FASPR, shape complementarity through a bundled `sc` binary, all behind the entry script's `--no-pyrosetta` flag. The catalogue leaves the pinned fork licence and parameter terms unverified; the commercial gate refuses it until that evidence is settled. One design carries a structure and a sequence, so the arm needs no sequence-design stage after it. `claude_binder.adapters.freebindcraft_generator` ships the hosted route and `rfdiffusion3-two-arm.template.json` binds it. The local route is deliberately absent: it needs a checkout, a GPU and a 5.6 GB parameter archive on the machine running the stage, and no check in this package can confirm any of the three.

| Item | Value |
| --- | --- |
| Source pin the recorded route served | `8b8d4c4627da06f084f46b7007450048fe3ca22f` |
| Weight layer | `alphafold_params_2022-12-06.tar`, 5,587,968,000 bytes, fetched by the deployment at setup and never pinned here |
| Recorded hardware | GPU H100, one runner, Python 3.12 |
| Per-request design ceiling | 32, with an answer ceiling of 256 staged designs |
| Seed | The served contract carries no seed field |
| Claim level | `contract-test` |

The parser fixture in `test_freebindcraft_generator.py` follows the application's writer. Qualify a selected deployment with one design and inspect its sequence and full-atom pose before a batch.

**The tool's accept verdict is not a retention gate on this path.** Behind `--no-pyrosetta` the fork returns a fixed number for eight interface fields, and four of them carry an active threshold in the fork's own defaults that each constant passes. The adapter therefore publishes a staging set: every design that reached a relaxed best model, whether the tool kept it or dropped it, with `export_set` staging, the verdict at `retention`, and the filter columns behind a drop. No row of that set may be published as an accepted binder. `rejected_only_by_placeholder_filters` marks a rejection the shipped thresholds cannot produce, which means the filters in use tightened one past its constant.

**Chain B is reserved.** The application returns the binder on chain B and refuses a request whose target chain is B. A campaign designing its binder on B takes the pose through unrelabelled; a campaign on another letter has the two letters swapped in column 22 of the PDB pose. `campaign.template.json` and `campaign.example.json` declare target B and binder A, which this route refuses by name; the TSLP demo and both local-contract fixtures declare target A and binder B, which it serves unchanged.

The 50-backbone roster floor needs two requests, 32 and 18. Each carries its own request id and its own `-b<NNN>` binder-name suffix, because the deployment keys its persistent run directory on both and builds every design name from the binder name.

Use the target's control panel and repeated seeds to compare PXDesign and Proteina-Complexa. A single candidate at one seed does not establish a ranking between generators.

PXDesign and Proteina-Complexa have adapter contracts in `rfdiffusion3-two-arm.template.json`. Neither has a bring-up PASS row, because neither has run under this package's contract.

### BoltzGen bound on 2026-09-12

BoltzGen served for the first time on 2026-08-28 through a hosted profile that closes at `contract-test` claim level, and that profile declared no production scoring. It now has a catalogue entry and an adapter module this package ships, `claude_binder.adapters.boltzgen_generator`, bound in `rfdiffusion3-two-arm.template.json`. It has not run under this package's contract.

| Tool | Role | Pins | Claim level |
| --- | --- | --- | --- |
| BoltzGen | Sequence and structure co-design | Package `boltzgen` 0.3.2, protocol `peptide-anything`, `--use_kernels false`, weights downloaded at runtime and unpinned | `contract-test` |

**No upstream commit is pinned.** The catalogue row and the adapter contract both carry `__REQUIRED__` at `source_revision`, which is one field more than PXDesign and Proteina-Complexa leave to the operator. The gap is upstream's rather than this package's: no file readable from here records a BoltzGen repository revision, so there is no value to withhold. The operator's own deployment reports the revision it built from, and the adapter records what the runner answers.

Keep BoltzGen's native filter result beside independent co-fold scores. One predictor seed does not overrule the generator's own filter. The adapter writes the native metric record to `<phase>/metrics/<candidate_id>.json` and carries `native_filter_state` on every manifest row. When a service returns no explicit filter verdict, the adapter records `unreported` instead of inferring a pass from metric names.

**BoltzGen 0.3.2 has no CLI seed control.** The proven invocation exposes no seed flag, so a receipt records `seed_delivered: false` and `used_seed: null`. Two consequences follow. A row cannot reproduce its own design. And splitting a phase across requests cannot vary a seed, so every request in a split phase carries the same requested seed and the requests differ only by BoltzGen's own sampling. Proteina-Complexa accepts and reports a seed, and PXDesign exposes none either.

**The served application caps one request at eight designs.** The published roster asks each method for fifty scored backbones, so the adapter dispatches a fifty-design phase as seven whole requests of eight, eight, eight, eight, eight, eight and two. Each request carries its own request id, its own output directory and its own receipt, and every row records which one produced it at `dispatch_batch`.

The verified BoltzGen adapter route accepts PDB input. Target preparation already emits that format. Direct CIF submission is refused before dispatch because the service converts it with Gemmi, and this stdlib adapter cannot reproduce the converted input digest locally. This is a boundary of the package route, not a limitation of the upstream application.

**The adapter checks what the application echoes.** The service fixes `peptide-anything` internally and does not return a protocol field. A response is refused unless it echoes the SHA-256 of the target structure the request sent, echoes the hotspot residue list the request sent, maps one chain position per requested residue, and names a checkpoint digest. A design built against a different target or a different site is not a design for the campaign that asked for it.

**The code is MIT and the weight terms are unverified.** The upstream README downloads gated model files without stating separate weight terms, and no file here quotes either licence at a pinned revision. The gate refuses a commercial campaign that selects BoltzGen on that ground.

### SolubleCaliby bound on 2026-09-13

`claude_binder.adapters.solublecaliby_designer` binds fixed-backbone sequence
design on the expanded profile's RFdiffusion arm. It checks the actual clean
Caliby checkout against the pinned source revision, hashes the supplied local
checkpoint, fixes whole non-design chains with Caliby's chain selector, and
parses the native CIF and designed sequence before publishing a candidate.
The observed checkout revision and checkpoint digest travel with the output.

Supply a compatible interpreter, checkout, checkpoint file and SHA-256.
[The pinned source and model-card reading](evidence/solublecaliby-upstream-2026-09-13.md)
records the Apache-2.0 terms and the `soluble_caliby_v1` checkpoint choice.
No completed local model run, hardware measurement or billed cost is claimed.

The adapter also exposes separate ensemble generation and ensemble design
commands. The expanded profile selects fixed-backbone design only. A published
32-structure Protpardelle-1c ensemble requires its additional dependencies,
weights and qualification; binding this designer alone does not reproduce that
experiment.

### BindCraft2 bound on 2026-09-21

BindCraft2 is the Pacesa Lab's second-generation binder designer at `PacesaLab/BindCraft2`. It
optimizes a sequence through AlphaFold 2 on the ColabDesign framework, redesigns with its own JAX
ProteinMPNN, and validates with separate AlphaFold models and structural filters. One design
carries a structure and a sequence, so the arm needs no sequence-design stage after it.
`claude_binder.adapters.bindcraft2_generator` runs a process inside a user-controlled
worker, and `bindcraft2-local.template.json` binds it. The
[2026-09-21 Modal run](current-status.md#bindcraft2-on-modal-2026-09-21)
verified the toolcheck, upstream execution, and package parser. Full Binder
execution and the downstream screen remain unverified. Settled dollars are pending.

When the user leaves compute open and Modal is connected, prepare the packaged
[Modal image](../envs/bindcraft2_gpu.py). Use the connected Modal interface for
image setup, persistent storage, submission, and artifact retrieval; reuse
matching compute notes when they exist. The recipe
provides the source pin and installation layout. Resolve the executable in that
image, derive environment identity from the built environment, and record model
identity from checkpoint hashes. Fill those profile values yourself. Keep paid
setup and execution within the approved scope.

The local profile also selects an ESMFold2-Fast screen. The BindCraft2 image
contains the designer's dependencies; prepare a compatible screen environment
separately. Use the native compute interface when that is the available route,
then retain parsed artifacts for the next stage. BindCraft2's own accepted
output alone does not establish a completed Binder screen.

The licence permits private-cloud use and user-installed copies directed by
the user's own tooling. It restricts providing the software's functionality to
third parties as a hosted service. Those conditions apply on both Modal and fal.
The process adapter has no HTTP transport, so endpoint flags belong to its outer
worker transport. That interface limit does not prohibit private cloud use.
Read [tool licences](tool-licences.md) for the terms and the
[fal route](fal-route.md) when that provider is requested.

| Item | Value |
| --- | --- |
| Source read | `18a9042fbe9a5373a5b7d98fe82335127c2fd70d`, five commits past `v1.0.0`, read 2026-09-21. Upstream tagged `v1.0.1` at `5342aefa18dedad653f7a5f6dbee1e566ca24d8f` and it was HEAD when checked later that day. Nothing here has been re-read at that tag, and the twelve modality presets, the four binder scaffolds and the four presets the multi-chain pre-flight reads are byte-identical at both |
| Declared version | `bindcraft/campaign_log.py` sets `VERSION = 'BindCraft 2 v1.0.0'` |
| Shipped weights | Twelve ProteinMPNN and HyperMPNN checkpoints, 6,681,030 bytes each, under `bindcraft/weights/proteinmpnn/weights_{neutral,negative,positive}/v_48_{002,010,020,030}.npz`; default model `v_48_020`, default variant `negative` |
| Fetched weights | `alphafold_params_2022-12-06.tar`, stated 5.3 GB, from `storage.googleapis.com/alphafold`, on first use or by `bindcraft fetch-weights`; seven files are needed, `params_model_{1..5}_multimer_v3.npz` plus `params_model_1_ptm.npz` and `params_model_2_ptm.npz` |
| Environment | Python 3.12 or newer, `jax>=0.11,<0.12`, accelerator extras `cuda12`, `cuda13`, `rocm`, `oneapi`; `install.sh` refuses a CPU installation, picks the extra from the CUDA major version `nvidia-smi` reports, and demotes `cuda13` to `cuda12` when a visible card reports compute capability below 7.5 |
| Reproducible mode | `--core benchmark` pins `campaign_seed` 0 and turns off `autotune` and `desperation` |
| Claim level | `unqualified` for the complete Binder arm; toolcheck, upstream execution, and parsing verified on Modal |

It answers its own vocabulary for free, and the adapter validates against that rather than a
table it carries. `bindcraft design` accepts `--list-targets`, `--list-modalities`,
`--list-properties`, `--list-core` and `--list-settings`, each printing one name a line. The
upstream package root imports only the standard library and resolves submodules lazily, the
dispatcher answers every `--list-*` flag before it imports the campaign, and `--list-settings`
reads the setting names out of the module source with `ast`. So none of the five loads jax, touches
a GPU or opens a socket. That was checked rather than inferred: all five ran against the checkout
on a macOS machine with no jax installed and no GPU, on CPython 3.14.7 with only the standard
library, and `--list-settings` returned 134 names with `jax` absent from `sys.modules` afterwards.
The adapter's toolcheck reads all five off the installed copy, and `run`
refuses an unknown modality, property, core profile or `--set` key by name before the subprocess
starts. That is the check that stops a campaign dying on a typo after it has paid for its
imports and its model load. The route has now been timed once, on 2026-09-21: a two-trajectory
campaign against upstream's own `hPDL1` target took 619.0 seconds on a Modal A100-80GB, and a
repeat of the same settings took 370.2 seconds with the jax compilation cache already warm. The
adapter's own toolcheck took 17.3 seconds on the same card and reported AlphaFold parameters 7 of
7 and ProteinMPNN checkpoints 12 of 12. The accepted design scored `i_pDAE` 0.31 with a hotspot
contact fraction of 1.0, and the same settings reproduced the same design in a second container.

Provenance arrives in the output rather than as an operator value.
`campaign_metadata.json` records the resolved settings, the source revision with `-dirty` appended
when the tree was modified, and SHA-256 checkpoint hashes. Most rows in this catalogue leave the
model revision `__REQUIRED__` because the tool does not say what ran. This one says, and the parse
stage carries it into the receipt.

**The whole profile runs without a provider endpoint, and that took one rebinding.** The overlay
inherits `small-run`, which overrides the ESMFold2-Fast predictor onto
`claude_binder.adapters.fal_esmfold2_fast_predictor` and carries `{{esmfold2_fast_fal_url}}` in its
argv and its cost basis. Composition reads the adapter list rather than the stage list, so removing
the RFdiffusion3 and ProteinMPNN adapters left that one endpoint required and a BindCraft2-only
campaign could not compose without fal. `bindcraft2-local.template.json` now restates the same
adapter id on the local route `full-ensemble` already ships, whose toolcheck is `pip show esm` and
whose argv carries no endpoint. Checked by composing both profiles: `small-run` requires
`esmfold2_fast_fal_url`, `proteinmpnn_fal_url` and `rfdiffusion3_fal_url`, and this profile requires
none. That route has executed: the 2026-09-02 Modal v14 shard calls
`claude_binder.adapters.esmfold2_fast_predictor` with no fal module anywhere in the shard, and its
receipt reports `ok` true with an empty error list. It settled at 0.02712564 USD per fold on a Modal
L40S against the fal route's modeled 0.61573375, and this profile still carries no price, because a
rate measured on an L40S does not predict the operator's own card. The profile takes that run's
`source_revision` and checkpoint pin, which are identities of the tool rather than of the machine,
and leaves `environment_identity` for the operator.

**The run names `--modality binder` on purpose, and that changes the campaign.** Upstream applies no
modality layer when none is named, and `binder` is in `BINDER_FORMATS`, so naming it reads
`settings/modality/binder.json` as a layer. That preset bans cysteine through `aa_bias`, sets
`min_monomer_plddt_final` to 0.7, and adds a `Binder_RMSD` filter at 3.5. The composed
`binder_lengths` win over the preset's own 60 to 180 range, because the request is the last layer
upstream applies. Omitting the flag and naming it are two different campaigns, so the stage
description records which one ships.

**`i_pDAE` is this package's own ipSAE with two ingredients swapped.** BindCraft2 ranks on
`i_pDAE`, and `bindcraft/filters.py` builds it like this: select the binder-to-target residue pairs
whose C-alpha atoms sit within 8 angstroms, take d0 from each residue's own partner count, score
each selected pair with `1 / (1 + (PAE / d0) ** 2)`, average over that residue's selected partners,
take the maximum over residues, and take the maximum over the two directions. This package's own
ipSAE at `binder_metrics._directional_ipsae` is the same construction, and says so: Dunbrack 2025
Equation 14 in the d0res variant, the same `1.24 * cbrt(n - 15) - 1.8` fit floored at 1.0, the same
kernel, the same per-residue average over the selected partners, the same maximum over residues.

Four things differ, and only the first changes the estimator. Dunbrack selects the pairs by PAE
below a cutoff, which this package sets at 10 angstroms, where BindCraft2 selects them by an 8
angstrom C-alpha contact. Those are different sets rather than the same set described two ways: a
partner with low PAE outside the 8 angstrom shell moves ipSAE and leaves `i_pDAE` unchanged, and a
contact whose PAE sits at or above the ipSAE cutoff moves `i_pDAE` and is left out of ipSAE. Because
d0 is taken from the selected-partner count, the distance scale moves with the mask.

Two more are smaller. `i_pDAE` reads the coordinates and the resolved-atom flags, so it is a
geometry-gated PAE score, where the lane's ipSAE reads the PAE matrix and no distance at all. And
`i_pDAE` pools every binder chain into one block against one target, where the lane scores one
ordered chain pair at a time.

The fourth changes the reduced value. BindCraft2 returns one binder-to-target PAE block and its transpose, so its
two directions are the same numbers read two ways. `compute_ipsae` slices `matrix[target, binder]`
and `matrix[binder, target]` independently, and AlphaFold PAE is asymmetric, so the lane's two
directions are two different measurements. `ipsae_min` is the smaller of two measurements where
`i_pDAE` is the larger of one measurement seen from both sides. The shared construction holds per
direction and not for the reduced value.

The d0 curves are the same curve. BindCraft2 clamps the partner count at 19 before the cube root
and floors the result at 1.0; the lane's `_d0` floors only the result. Evaluated against each other
they are bit-identical at 45 of the 60 partner counts from 1 to 60 and differ by one unit in the last place at the other 15, because upstream raises to `** (1 / 3)` where the lane calls `np.cbrt`, with a largest gap of 8.9e-16. d0 stays pinned at its 1.0 floor until a residue has 27 partners. So on an ordinary binder interface both reduce to the same
fixed `1 / (1 + PAE ** 2)` kernel and the pair mask is the whole difference. A second reading of
Dunbrack's own `ipsae.py` reports that its `calc_d0_array` raises the count to 26 rather than 19 and
reaches the same values for the same reason, which is consistent and has not been checked here
against the script.

**So `ipsae-min` is not a construction-independent second opinion on a BindCraft2 row, and the
screen is still worth running.** What makes it a real check is the predictor rather than the
formula. The lane refolds the candidate with its own co-folding arm, and
[co-fold scoring hazards](cofold-scoring-hazards.md#a-generators-own-ranking-score-can-be-the-screening-metric-under-a-different-pair-mask)
carries the full reading, including the measurement that the predictor dominates: over eight
designs, the same ipSAE implementation on two predictors ranked them at Spearman -0.1429.
`minimum_sc_dockq_ensemble` and the clash ceiling are a different construction, though not an
unrelated one, since both read predicted geometry. What has not been shown is that BindCraft2's
selection on `i_pDAE` actually transferred to the lane's screen, and nothing here has measured it.
Read a passing `ipsae-min` on such a row carefully rather than discounting it.

**Acceptance is the tool's own computational verdict and it is a real gate here.** Unlike the
FreeBindCraft path, no threshold compares a constant: PyRosetta is absent from the upstream tree
entirely, so there is no `--no-pyrosetta` substitution and nothing goes inert. `3_Ranked/!_Ranked.csv`
is the single record of what was accepted, best-first by `i_pDAE`. That still measures prediction
confidence and geometry rather than binding, so rank on an independent screen and publish no
accepted row as a validated binder.

It can run with no network at job time, which is rare in this catalogue. Once the parameters
are on the machine a campaign reaches no host, and the upstream ships `bindcraft fetch-weights` for
a compute node with no route to the internet. A first run with no parameters present downloads
5.3 GB instead. The adapter's toolcheck reports a missing parameter set as a named missing input
and fetches nothing.

**There is no BindCraft2 paper to cite, and the one to read is BindCraft's.** No DOI, preprint or
citation for BindCraft2 appears anywhere in the upstream checkout. The design guide points at the
original BindCraft wiki and says the biology intuition carries over while the settings, modalities
and outputs are BC2's, and nothing in the tree states what changed between the two. The first
generation is published: Pacesa, Nickel, Schellhaas and others, "One-shot design of functional
protein binders with BindCraft", Nature, 2025, DOI `10.1038/s41586-025-09429-6`, confirmed against
Crossref on 2026-09-21. Read that for the method and treat nothing in it as measured evidence about
this tool.

The packaged Modal A100-80GB recipe has a recorded BindCraft2 toolcheck and
upstream run with parsed ranked output. Its [run record](current-status.md#bindcraft2-on-modal-2026-09-21)
gives durations. No settled per-job charge is attached to that run; estimate
another campaign from its own rate, wall-time limit, and receipts.

## Per-tool notes

These notes give the input and output contracts that matter when connecting a
tool to the campaign.

### Genie3 wiring contract

Genie3 in binder mode generates a C-alpha trace containing only `CA` atoms without `N`, `C`, or `O` atoms, as declared for `genie3` in [catalog.json](../claude_binder/data/catalog.json). ProteinMPNN must receive that trace through its C-alpha mode, which selects C-alpha weights and the matching parser. Vanilla ProteinMPNN fabricates missing `N`, `C`, and `O` atoms, producing sequences that describe the fabrication instead of the generated fold.

The shape guard enforces atom compatibility at two points. `route_problems` compares catalog `produces` and `requires` sets, follows a fixed `--backbone-stage-id` before dependency order, and reports a missing atom only when both declarations are known, as implemented in [backbone_shape.py](../claude_binder/backbone_shape.py).

`pose_shape_problem` reads PDB or mmCIF atom records, samples the first 4 residues per chain, and scans up to 50,000 atom records before returning a missing-atom problem. The ProteinMPNN adapter calls that check before its first design call and raises on a short chain in [proteinmpnn_designer.py](../claude_binder/adapters/proteinmpnn_designer.py). The guard reports or refuses an incompatible shape. It does not select `--ca-only`, C-alpha weights, or a parser. The wrapper now carries that route: `--ca-only` in [proteinmpnn_designer.py](../claude_binder/adapters/proteinmpnn_designer.py) loads the `ca_model_weights` checkpoint family and passes `--ca_only` to the runner, and `full-ensemble.template.json` selects it through the `proteinmpnn-designer-genie3` adapter, which reads `--backbone-stage-id generate-genie3`.

The invocation contract uses `genie3 generate`; `genie3 run` also enters evaluation, whose recorded ProteinMPNN resolution fails. Set `cwd` to the weights directory because Genie3 resolves `pretrained/v1/config.yaml` relative to `cwd`.

The input contract uses this binderbench layout:

```text
problems/<selection>.json
targets/pdb/<selection>{,-chain_X}.pdb
targets/fasta/<selection>{,-chain_X}.fasta
targets/msa/<selection>{,-chain_X}.a3m
```

Write absolute paths into `problems/<selection>.json`, including `target_pdb_filepath`, because Genie3 resolves that field from `cwd` instead of the dataset root.

The generated PDB contract writes to `<rootdir>/<selection>/pdbs/<selection>_<sample_idx>.pdb`. The parser maps those files into the profile's `candidate-manifest`, `sequence-files`, and `design-pose-files` artifacts, preserves file hashes, and writes `{{attempt_dir}}/{{phase}}/parser-result.json`, as required by [full-ensemble.template.json](../claude_binder/data/templates/profiles/full-ensemble.template.json).

The environment contract separates what `genie3 generate` needs from what the recorded RunPod install carried. The shipped recipe is [genie3_generator_gpu.py](../envs/genie3_generator_gpu.py).

The Hardware cell in the generators table pairs two figures that come from different files. 24 GB is `resources.gpu_memory_gb` in the `genie3-generator` contract, copied into `catalog.json` as `hardware.gpu_memory_gb`. L40S is `gpu_default` in the recipe, which calls it the smallest recorded card that clears that declaration, at 48 GB. TODO: `catalog.json` leaves `hardware.recommended_gpu` for `genie3` as `__REQUIRED__`, so no catalogue field names the card and the two halves of that cell cannot be reproduced from one source.

1. **Base:** `torch==2.7.1` from the cu126 wheel index. cu126 is the index that carries the pinned version, because cu124 holds nothing above `2.6.0+cu124`, and a served binder-mode run reported `2.7.1+cu126` with CUDA runtime 12.6. The recipe pins no CUDA base image, so no base release can disagree with the wheel.
2. **Numpy and Cython:** `numpy>=2.0.2,<3` and Cython, installed before the source build, which resolves biopython itself.
3. **Torchvision and ColabFold parameters:** `torchvision==0.22.1 --no-deps` and `GENIE3_ALLOW_COLABFOLD_PARAMS=1` belong to the recorded install, which built the `genie3 evaluate` stack on a pod shared with Boltz. The shipped recipe skips that stack and carries neither.
4. **Boltz handoff:** `numpy<2.2` immediately before Boltz, because Boltz's numba dependency caps numpy at 2.1. That pin repairs Boltz on a shared pod and generation never reads it.

The weights host is `yeqinglin/genie3` at revision `9ae31ebb8c56eebdc05ab282a8fd3f6a6d2a03a2`. The `genie3-generator` contract in [full-ensemble.template.json](../claude_binder/data/templates/profiles/full-ensemble.template.json) carries that revision as `model_revision`, and [genie3_generator_gpu.py](../envs/genie3_generator_gpu.py) fetches the same revision and refuses to hydrate unless the `pretrained/` subtree comes to 4 files and 536135478 bytes. Verify the adapter receipt against that revision before any scored run.

The code licence is Apache-2.0, with commercial use permitted in [tool-licences.md](tool-licences.md). The weights carry Apache-2.0 as well, declared as model card metadata at `yeqinglin/genie3` and not in a licence file, because that repository holds none. [catalog.json](../claude_binder/data/catalog.json) records the declaration with commercial use permitted.

Full-backbone ProteinMPNN Exact requires N/CA/C/O backbones, such as qualified
RFdiffusion3 or PXDesign outputs. Genie3's C-alpha binder output instead uses
the documented compatible C-alpha designer route. Its [native kit](genie3-kit-modal.md)
has passed Modal inference and remains available for the user's selected
workflow.

For the fixed `genie3-generator` graph contract, the operator still resolves
`environment_identity` and `resources.container_image_digest`. Its command
line was written into the three argv templates on 2026-09-12 from the generic
Modal recipe and the adapter's argparse. Its compatible C-alpha designer route,
revisions, weights licence and environment are recorded separately from the
native acceleration-kit qualification. A native kit run does not wait for
that graph binding; use the [integration guide](accelerated-kit-integration.md)
to prepare its environment and explicit artifact handoff.

### The profile id does not say which RFdiffusion you get

`catalog.json` carries two entries, `rfdiffusion` and `rfdiffusion3`, and both record the same selection: `profile_id_value: "rfdiffusion"` in `generation.generators` with `adapter_id: "rfdiffusion-generator"`. Writing `"id": "rfdiffusion"` does not distinguish RFdiffusion from RFdiffusion3. The adapter contract's `model_revision` and checkpoint specify which model executes. Report that field to the scientist and record it with results because the two models differ.

Two modules implement this generator. `claude_binder.adapters.rfdiffusion_generator` declares `DEFAULT_ADAPTER_ID = "rfdiffusion-generator"` and serves shipped profiles. A Modal campaign uses this adapter's `modal` runner protocol. Read [Modal operations](modal-bring-up.md) for the required per-environment values.

`claude_binder/adapters/rfdiffusion_generator.py` declares `RUNNER_PROTOCOLS = ("auto", "local", "modal")`, and `modal-first-run-floor.json` binds it to Modal. That canary profile leaves its environment hash as `__REQUIRED__`. The historical H100 measurement was taken in a hosted environment on another provider, so its environment identity does not name a Modal environment and cannot be reused as one. Disclose the true compute provider on every approval card and charge all stages against the campaign budget.

### A hosted generator can swap the chains

A hosted Genie3 run designated chain A as the 95-residue binder and chain B as the 115-residue target. The adapter published chain B as the binder, causing ProteinMPNN to redesign the target. Check residue counts against campaign binder length bounds before publishing a chain.

### RFdiffusion3 needs strict PDB and a real ligand code

The input parser requires standard 80-column PDB lines with element symbols in columns 77 and 78. Use the exact Chemical Component Dictionary three-letter code for ligands. Generic placeholders such as `LIG` cause atom-count mismatches. Set `allow_ligand_on_existing_chain: true` when the ligand is pre-positioned on an existing chain.

### ESMFold2 has a package-identity trap and a size limit

The `esm` package on PyPI is a separate project. ESMFold2 arms require `esm` from `github.com/Biohub/esm` on Python 3.12, with `requires-python = ">=3.12,<3.13"`. It cannot share an image with a Python 3.11 stack. ESMFold2-Fast depends on the ESMC-6B backbone, bringing total weights to 26.2 GB.

ESMFold2 accepts CCD-coded ligands only. It rejects arbitrary SMILES or SDF input.

On protein-RNA complexes, ESMFold2 misplaces RNA coordinates in both model sizes. Full scored DockQ 0.009 against Fast at 0.015, while Boltz-2 on the same case scored 0.415. Use an AF3-class co-folder for protein-RNA interfaces.

### Residue numbering drifts between the target and the prediction

ESMFold2 target chains frequently number from 1 even when the target window spans residues 29 to 128. Align structures by ordered C-alpha coordinates rather than residue index integers. Unaligned RMSD between independent prediction runs produced a spurious 43 angstrom measurement, confirming that Kabsch superposition is required before calculating RMSD.

### A single predictor's confidence is not evidence of binding

Scrambled-sequence controls scored Boltz iPTM between 0.80 and 0.89 and ipSAE up to 0.65. Of five top Genie3 designs selected with Boltz iPTM between 0.85 and 0.92, zero survived AF2-Multimer cross-evaluation, scoring 0.10 to 0.24. Evaluate designs with an independent co-folder outside the generative model family, and run scrambled controls.

### FreeBindCraft's own filters go inert without PyRosetta

When running with `--no-pyrosetta`, eight interface fields return constant values and four thresholded filters stop discriminating. That configuration dropped 31 valid candidate sequences. Export the staging set with `--stage-all` and screen candidates independently.

### Boltz and ESMFold2 write mmCIF into nested directories

Boltz 2.2 writes outputs to `boltz_results_<stem>/predictions/<stem>/`. Genie3 writes to `<rootdir>/<selection>/pdbs/<selection>_<sample_idx>.pdb`. Parsers expecting flat PDB directories fail to find outputs.

### DockQ is undefined, not zero, without reference contacts

When a pose carries declared chains but no reference contact residues, DockQ, Fnat, iRMSD, and LRMSD are mathematically undefined. Record `scored-degraded` with null reference metrics. Assigning zero creates false ranking penalties.

## Provider limits you will hit

These observations describe recorded deployments. Read the selected endpoint's
current limits and price its actual runtime before choosing deadlines or widths.

A recorded hosted route had a 7,200-second request limit and rejected a
14,100-second deployment. For that limit, a 6,600-second child deadline reserves
600 seconds for evidence serialization. Use the selected route's limit for a
different deployment. Split longer work into resumable shards with persistent
storage and retain the progress ledger for timeout diagnosis.

`max_mpnn_sequences` sets the number of passing MPNN sequences to validate inside
a trajectory. Use 1 when the shard needs one passing sequence. Preserve a larger
scientific cohort when requested, and budget or shard the additional validation.
For wide sites, bound each shard by wall time and record attempts, including
shards with no accepted design. `max_mpnn_sequences` controls validation work;
it does not set a universal acceptance rate.

The artifact contract requires completed disk writes rather than an immediate HTTP response. A recover-only probe can encounter an incomplete pre-serialization state while the attached job completes successfully. Store long design jobs in persistent directories so that timeouts do not purge output FASTA and pose files.

Include model installation and weight hydration in a cold-start estimate. [Measured
costs](measured-costs.md#make-the-cost-decision) describes how to price the selected route.

## Choosing a stack

Claude Science recommends a stack from the campaign requirements, live routes,
cost evidence, and tool records. The package executes the selected profile; it
does not supply a universal tool ranking. Treat profile defaults as a starting
configuration that the scientist can revise.

Three stances are available:

**Reproduce the published stack.** Match the protocol's tools, versions, targets,
constructs, controls, scale, seeds, and scoring rules. Use the
[published campaign comparison](published-campaign-comparison.md) to record
departures. Matching tool names alone establishes only tool overlap.

**Substitute inside a stage.** Keep the pipeline structure and replace a selected
tool. Examples include Chai-1 as a secondary co-folder or SolubleMPNN for a
sequence-design arm. Before paid Binder execution, establish the selected
adapter's input, output, runtime, authorization, and cost contract. A tool with
its own Claude Science runbook can execute standalone and return hashed
artifacts through a handoff. Exploring or recommending a substitution requires
no package bring-up PASS.

**Change the shape.** ESMFold2 native design optimizes backbone and sequence together, eliminating separate generator and designer stages. It can aim at a designated epitope. The adapter takes `epitope_residues` and builds an inter-chain contact mask over those positions, as implemented in [constraints.py](../claude_binder/native_design/constraints.py). The adapter refuses an unaimed run until the caller passes `--acknowledge-unaimed`, in [esmfold2_native_design.py](../claude_binder/adapters/esmfold2_native_design.py). An aimed run carries the adapter's own status string `implemented-unvalidated`, which states the position exactly. The code path exists and no completed campaign has measured it.

Choose scoring arms for the question being tested. Same-family scoring can
screen candidates or compare settings. For an out-of-family confirmation,
select a co-folder outside the designer's family and record model, weights,
alignment, and seed differences. ESMFold2-Full and ESMFold2-Fast share an
embedding lineage. Different model names or providers alone do not establish
statistical independence. A profile that claims a particular confirmation
standard must retain the arms and controls required by that standard.

For cost projections, see [measured costs](measured-costs.md). For campaign rounds, see decision 8 in [the nine decisions](the-nine-decisions.md); decision 6 selects the metric the rounds optimize.

### A worked plan

A campaign asks for two rounds on a target, a 300 dollar budget ceiling, and an alternative generator.

1. **Route the tools.** ESMFold2-Full and ESMFold2-Fast have deployed screening
   records. For out-of-family rescoring, compare the selected co-folders and their
   live rates. Boltz Cloud has a package adapter. Chai-1 and OpenFold3 can use
   their native runbooks and artifact handoffs; direct Binder execution requires
   an adapter contract.
2. **Route the generator.** `rfdiffusion-generator` is the active wired generator. Genie3 is wired in `full-ensemble.template.json` and requires an `environment_identity` and a container image digest before materialization passes.
3. **Size the fold count.** `lane.estimate_fanout` counts generation, controls, predictor arms, seeds, optimization, and smoke calls from the materialized plan. For the shipped Modal profile at 10 selected candidates, it counts 84 co-folds: 14 screen and 70 rescore. The simpler `claude_binder.cost` calculation omits controls and smoke calls. Use the graph estimate for admission control. [Cost details](measured-costs-detail.md#calculate-the-example-fanout) give the reproducing command.
4. **Price it.** Read the selected provider's rate for the chosen hardware, storage, and job lifecycle. Combine that rate with bounded wall time for the graph's jobs, including setup and retries. Set `provider.budget.maximum_spend_usd` above the materialized estimate and within the scientist's approved ceiling. After one handoff, update the next estimate from its measured receipt. [Measured costs](measured-costs.md#make-the-cost-decision) explains the accounting.
5. **Check ceiling enforcement.** No standing provider ceiling ships. `STANDING_CEILING_USD` in `run_intent.py` is unset for every provider, so the campaign ceiling you set is the only dollar cap unless the installation configures an account-policy record, in which case the lower of the two binds. Check for that record before quoting a number. See [ceiling mechanics](measured-costs-detail.md#the-ceiling-the-code-enforces-today).
6. **Set the stopping rule.** Rounds terminate upon meeting six conditions, including improvement falling below `optimization.early_stop_margin` or reaching `optimization.rounds`. Implemented optimization operators include `point-mutation` and `inverse-folding-resample`. Selecting `partial-diffusion` or `predict-redesign` causes controller errors.
7. **Name the metric.** `scoring.primary_metric` designates the ranking objective, with `ipsae_min` serving as the interface metric used in the published protocol. Pair it with `sc_dockq` for pose geometry. Cross-evaluate survivors with a different predictor family and run shuffled-sequence controls through the same settings. A high score alone does not verify binding specificity.

## Open gaps in this catalogue

Five tools carry an unbound status: `chai1`, `openfold3`, `fair-esm2`, `ncbi-sequence-fetch`, and `diffdock`. Each records its evidence inline. Four cite the 2026-08-29 platform snapshot tagged `PLATFORM SNAPSHOT 0.1.41-release` in the tool's `availability.evidence`. `ncbi-sequence-fetch` is an optional sequence-retrieval capability rather than a bundled platform skill. Discover a compatible route in the current session before selecting it. Every other catalogue entry names an explicit status. `operator_configured` rows are package contracts with concrete operator prerequisites. They do not claim that a route is live or scientifically qualified.

The pinned MMseqs2 installation, UniRef90 release, immutable bundle checker, no-egress contract, and remaining runtime integration work are recorded in [MMseqs2 + UniRef90 local route](mmseqs2-uniref90-local-route.md). A full-database capacity canary is still required before promotion.

The pinned Protenix v2 container and hydration design are recorded in [Protenix v2 bring-up](protenix-v2-bringup.md). Eighteen of the nineteen objects the pinned source lists answered a one-byte range request on 2026-09-14, and only the v2 checkpoint was denied, so that page also sets out three ways to reach a checkpoint and what each one supports. The official checkpoint receipt and GPU qualification remain open.

The full-ensemble profile names the seven published methods and carries no anonymous co-design slot, so it cannot claim baseline fidelity from an unnamed third identity.
