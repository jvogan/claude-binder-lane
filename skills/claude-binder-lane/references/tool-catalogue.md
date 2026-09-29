# Tool catalogue

Every tool a campaign calls runs through a named route. The route determines what you build, what you pay, and what failure modes can occur. Read this catalogue before making decision 4 in [the nine decisions](the-nine-decisions.md), because the tool-stack decision assumes you know which tools are reachable.

The recorded platform snapshot was read from Claude Science `0.1.41-release` on 2026-08-29. Live availability must be discovered through the session as described in [Discover tools and routes](platform-tool-discovery.md). Binding status comes from `claude_binder/data/catalog.json`. Detailed bring-up and integration records are in [Tool catalogue details](tool-catalogue-details.md); use them when a selected tool needs operator setup or a claim depends on its recorded behavior. Cost links describe Claude Binder evidence and planning limits, not runs from an earlier workflow.

## Contents

- [Derive the route matrix](#derive-the-route-matrix)
- [Route shapes](#route-shapes)
- [What Claude Science already ships](#what-claude-science-already-ships)
- [NVIDIA BioNeMo NIM](#nvidia-bionemo-nim)
- [NVIDIA BioNeMo Inference Runtime](#nvidia-bionemo-inference-runtime)
- [What the package binds](#what-the-package-binds)
- [Which profile to start from](#which-profile-to-start-from)
- [Which adapters are wired](#which-adapters-are-wired)
- [Tools you bring up yourself](#tools-you-bring-up-yourself)
- [Per-tool notes](#per-tool-notes)
- [Provider limits you will hit](#provider-limits-you-will-hit)
- [Choosing a stack](#choosing-a-stack)
- [Open gaps in this catalogue](#open-gaps-in-this-catalogue)
- [Tool catalogue details](tool-catalogue-details.md)

## Derive the route matrix

The prose below describes the route shapes a tool can occupy. To read which
routes a tool occupies today, derive it:

    PYTHONPATH="$BINDER_SKILL" python3 -B -m claude_binder.route_matrix --markdown

The module reads `claude_binder/data/catalog.json` and every profile template
through the loader the executor uses, then reports per tool: each profile that
enables it, the route class that profile runs it on, the provider that profile
names, whether the named adapter module exists, and any module that ships with
no profile binding it. Add `--platform-inventory <snapshot.json>` to join the
live Claude Science skill and compute inventory, and `--tool <id>` to narrow the
answer. Drop `--markdown` for the JSON record.

Five route classes, each stating what the tree proves:

- **local-process** means a profile runs the adapter as a local subprocess.
- **local-fixture** means the profile uses the deterministic fixture adapter and contacts no provider.
- **hosted-deployment** means the adapter argv takes a deployment URL the scientist owns, or names a `fal_` client module.
- **self-hosted-provider** means the profile declares a `provider` block, and the `providers` field names which one.
- **hosted-api** means the profile runs the adapter as a local subprocess but the catalogue records
  the tool reaching a named host at job time, so the work happens on the vendor's machine. Boltz is
  the case that named this class: its profile row looks local and its module binds the hosted
  Boltz Cloud API.

A route class is a planning fact. It does not establish qualification, a price,
licence clearance, or a scientific result.

**A `hosted-api` or `hosted-deployment` measurement does not transfer to self-hosting the same
software.** Running a model through a vendor's endpoint exercises the vendor's build, on the
vendor's hardware, at the vendor's settings. It tells you the wall time and the price of that
route. It tells you nothing about what the same software does on hardware you rent, because
changing route changes the build, the machine and often the sampling settings at once. Recording
that a tool has run does not record which route it ran on, so state the route whenever you cite a
result.

Three findings from the derivation on 2026-09-13, which the command reprints
rather than asking you to trust this page:

- 24 of the 32 catalogued tools have at least one route whose adapter module ships. Six have no such route: `boltz-local`, `chai1`, `diffdock`, `fair-esm2`, `ncbi-sequence-fetch`, and `openfold3`. Genie3 left that list on 2026-09-12 when its full-ensemble argv was filled in. That is a statement about what this package can dispatch, and it is not a statement that the tool is unreachable. Five of those rows carry `status: unbound` and a note saying the platform lists the tool while this package binds no adapter for it. `boltz-local` is the exception and carries `operator_configured`, because its `boltz2_local_predictor` module ships while no profile binds it. Claude Science can configure that profile and route, or use a platform skill or registered endpoint and preserve artifacts through [connector authoring](connector-authoring.md). OpenFold3 is the clearest case: it has no Binder adapter and a registered, exercised NVIDIA NIM endpoint, described in [managed model endpoint route](managed-endpoint-route.md). Two more, `chimerax` and `pymol-open-source`, have route rows only through the `viewer-renderer` adapter they share, and the catalogue records for each a local install the platform does not provide, so neither counts as runnable and each carries `runnable_route_blocked_by`.
- All three native providers are declared by a shipped profile. Modal has five: `modal-platform.template.json`, `small-run-modal.template.json`, `modal-first-run-floor.json`, `modal-smoke.json`, and `supplied-candidates-modal.template.json`. RunPod has `small-run-runpod.template.json` and Lambda Cloud has `small-run-lambda.template.json`, both added on 2026-09-12. The two new profiles run the same science as `small-run` on the operator's own container image, so every image digest, tool root and GPU type is `__REQUIRED__`, and all three inherited fal prices are dropped rather than reused on a different provider. Neither provider has run a job from this package, so both sit at provider-canary claim level and neither has a priced arm. Read [RunPod route](runpod-route.md) and [Lambda Cloud route](lambda-route.md) for what the lifecycle covers.
- Three modules ship in the adapters directory with no profile binding them: `fal_genie3_generator`, `chimerax_renderer`, and `boltz2_local_predictor`. The code is present and unreachable, which is a different answer from absent. `genie3_generator` was a fourth until 2026-09-12.

## Route shapes

| Route | What the route needs | What executes | Bring-up cost |
| --- | --- | --- | --- |
| Platform skill | The runbook's required local or remote runtime | Claude Science follows the visible runbook | Discovery is free; runtime bring-up depends on the selected tool |
| Package adapter | A profile that names a bound adapter | The package executor, which validates argv and parses the output | Adapter code ships; deployment and qualification may remain |
| Managed endpoint | A registered local or remote endpoint with its native payload | Claude Science calls the registered service | A free reachability probe, then an N=1 contract canary |
| Hosted model API | Provider credentials, a selected model revision, and its request and artifact contract | The provider's service through a model-specific client | A free account check where available, then an N=1 canary |
| Self-hosted image | A container image, pinned weights, and a driver script | The selected image through Modal, RunPod, Lambda, SSH, HPC, or another GPU host | Route-dependent. See [tool bring-up](tool-bringup.md) |
| Local | A local install | The stage on the session machine | Minutes for a CPU scorer |
| Manual handoff | Hashed outputs and provenance from another workflow | Binder imports artifacts at the next contract boundary | No adapter until the handoff is automated |

Modal, RunPod, and Lambda Cloud are distinct provider routes. Modal uses
`CLAUDE_BINDER_EXECUTION_ROUTE=modal-platform` and retains its separate
smoke/scale-wave contract. [RunPod](runpod-route.md) ships an HTTP client and
guarded lifecycle but needs a qualified Binder worker. [Lambda Cloud](lambda-route.md)
ships a lifecycle interface but no provider transport. Claude Science can use
either provider directly and can prepare an automated Binder route where the
actual provider contract supports it. A provider label alone does not establish
credentials, an endpoint, a worker, or a completed run. Provider-specific
account, image, instance, storage, rate, and usage evidence remains
provider-specific; evidence from one provider is never treated as evidence for
another.

A tool can occupy more than one route. ProteinMPNN ships as a platform skill, has a bound package adapter, and runs on a self-hosted CPU container. Boltz ships as a platform skill. The `claude_binder.adapters.boltz2_predictor` adapter binds only the hosted `boltz-cloud-cli` structure-and-binding route. Local open-source Boltz has its own shipped adapter, `claude_binder.adapters.boltz2_local_predictor`, declaring route `boltz-local-cli` at contract revision `local-predict-v1`. No shipped profile binds that route, so the profile binding and the runtime qualification are operator work, which is why `boltz-local` carries `operator_configured` above. NVIDIA Boltz2 NIM requires a separate endpoint contract.

A platform skill provides documentation and CLI recipes for Claude Science. In contrast, a package adapter provides a JSON contract specifying the role, entity types, artifacts, revisions, and argv templates that the executor runs with `shell=False`. Because execution requires this contract, a platform skill alone cannot place a tool into an executable run plan.

## What Claude Science already ships

The platform ships 29 skills. The table below is the subset a binder campaign calls directly. The
rest are listed under [the other shipped skills](#the-other-shipped-skills), so this page accounts
for all 29. The live skill catalogue is authoritative for the current session.

The set was re-read on 2026-09-05 against `0.1.43-release` and is the same 29 skills and 81 files as
the recorded `0.1.41` snapshot. Exactly two shipped files changed between those versions:
`remote-compute-modal/SKILL.md` grew from 49,756 to 50,550 bytes, and `self-awareness/SKILL.md` from
15,079 to 15,296. The Modal runbook is the one this package depends on, so treat any Modal note
written against `0.1.41` as needing a re-read.

| Skill | Role | Licence stated in the skill | GPU |
| --- | --- | --- | --- |
| `boltz` | Co-folding, protein, nucleic acid, ligand, optional affinity head | Apache-2.0 for the skill, MIT for Boltz-2 weights | Yes |
| `chai1` | Co-folding, second independent model for consensus | Apache-2.0 | Yes |
| `openfold3` | Co-folding, AF3-faithful settings | Apache-2.0 | Yes |
| `esmfold2` | Co-folding without an MSA, plus ESMC embeddings and mutation scoring | MIT weights on the `biohub` org | Yes |
| `alphafold2` | Folding and multimer prediction through the ColabFold runner | Apache-2.0 | Yes |
| `proteinmpnn` | Inverse folding | Apache-2.0 skill, MIT weights | No |
| `solublempnn` | Inverse folding biased to soluble expression | Apache-2.0 skill, MIT weights | No |
| `ligandmpnn` | Inverse folding with ligand, nucleic acid, and metal context | Apache-2.0 skill, MIT weights | No |
| `diffdock` | Blind small-molecule docking, geometry only | Apache-2.0 | Yes |
| `fair-esm2` | Embeddings, masked-LM likelihood, contact prediction | Apache-2.0 | Yes |
| `remote-compute-modal` | Modal dispatch | Platform | N/A |
| `remote-compute-ssh` | Optional platform route to a host you own; Binder's native RunPod/Lambda routes do not require it | Platform | N/A |
| `compute-env-setup` | Building a compute environment | Platform | N/A |
| `managed-model-endpoints`, `using-model-endpoint` | Calling a hosted endpoint. The registration and call contract is in [managed model endpoint route](managed-endpoint-route.md), and it is how a campaign reaches a model no adapter binds, including an NVIDIA BioNeMo NIM | Platform | N/A |
| `figure-composer`, `figure-style` | Figures, tiered: composer builds one multi-panel figure, style holds the rules | Platform | N/A |

### The other shipped skills

None of these is a binder tool. They are named so a reader can tell them apart from the ones above.

| Skill | What a binder campaign uses it for |
| --- | --- |
| `self-awareness` | `host.query()` over the session's own metadata for run history, cells, and artifacts |
| `borzoi`, `evo2` | Genomic sequence models, outside binder work |
| `scgpt`, `scvi-tools` | Single-cell expression models, outside binder work |

The platform ships further skills that fall outside computational structural biology. This package
neither uses nor describes them.

[Reading a finished run in Claude Science](reading-results-in-claude-science.md) says how the run's
own viewer, thumbnails, and sequence table hand off to the two figure skills, and what the platform
does not ship.

Platform skills define CLI recipes rather than a direct Python API. Claude Science reads the skill and executes the command in a compute environment. For example, Boltz takes a YAML complex definition and emits mmCIF files with pTM, ipTM, and pLDDT:

```bash
boltz predict complex.yaml \
    --use_msa_server --out_dir out/ --recycling_steps 3 --diffusion_samples 5
```

`--use_msa_server` transmits sequences to `api.colabfold.com`. Confirm that the scientist approves external transmission before using this flag on unpublished target sequences.

The recorded 0.1.41 snapshot contains no backbone-generator runbook. Discover
the live catalogue before selecting a route. RFdiffusion, RFdiffusion3, Genie3,
and BindCraft can use a package adapter, managed service, hosted API,
self-hosted image, local installation, or manual handoff when that route carries
the required model-specific contract.

## NVIDIA BioNeMo NIM

NVIDIA documents a Claude Science route that connects local or hosted BioNeMo
NIM endpoints. Its atomic skills cover MSA Search, RFdiffusion, ProteinMPNN,
OpenFold3, and Boltz-2. The route can therefore supply one stage or a complete
generation, design, and co-fold workflow.

NIM is an execution surface rather than a model family. Record the selected
model and route independently. A visible NIM runbook remains usable through
Claude Science when Binder marks the corresponding tool `unbound`; import its
hashed outputs through a manual handoff or supplied-candidate manifest. Read
[NVIDIA BioNeMo NIM route](bionemo-nim-route.md) for MSA topology, model-specific
request differences, local storage figures, and the artifact record.

## NVIDIA BioNeMo Inference Runtime

[BioNeMo Inference Runtime (BioIR)](https://github.com/NVIDIA-BioNeMo/BioNeMo-Inference-Runtime) is a self-hosted NVIDIA GPU inference runtime for structure predictors, including Boltz-2 and OpenFold3. Its Python `build_processor` pipeline accepts sequence and MSA requests, then writes CIF/PDB structures and score JSON. Use it to refold and score campaign candidates on a suitable GPU host; the [BioIR route guide](bioir-route.md) gives setup, request, and result handoffs. This documented native route sits outside the packaged adapter and `catalog.json` list.

BioIR Boltz-2 and an accelerated-kit Boltz-2 run count as the same predictor family. For a campaign requiring independent model families, use a distinct complex predictor such as OpenFold3 for the second prediction.

## What the package binds

`catalog.json` records 32 tools, and every entry carries one of five availability statuses.

**A status describes this package rather than Claude Science.** It answers one question: can a Binder campaign stage dispatch this tool on its own, so its output enters the same ranked table as every other arm? Claude Science ships skills of its own for several of these models, and you can run those in a session whatever this column says. The platform exposes `host.skills.list()` and `host.skills.read()` and no call that runs a skill, so a scientist follows one of those skills by hand while this package cannot invoke it. A tool you run that way still feeds the next Binder stage, by handing it the hashed artifacts. [Discovery](platform-tool-discovery.md) lists what your own session carries.

- **deployed**: an adapter contract exists and the tool has run.
- **operator_configured**: an adapter contract and profile route exist, and the pinned runtime must be supplied and validated before execution. That work is Claude Science's to carry out from [tool bring-up](tool-bringup.md), not a task for the scientist. The status records that the package does not ship the runtime, and it makes no claim about who stands it up.
- **unbound**: this package has no adapter contract for the tool, so no Binder stage dispatches it. Claude Science may still list the tool as a skill of its own, and running it there and handing on the artifacts is the supported route. Naming an unbound tool in a profile does not make a stage execute it.
- **refused**: the package recorded an architectural reason preventing the tool from running here.
- **conditional_local_install**: the package generates execution scripts, and you supply the local program installation.

One further value appears in the Status column and is not a catalogue status:

- **no entry**: `catalog.json` holds no entry for the tool. Nothing in that row was read from the catalogue, so each such row names its source below the table.

Reproduce the Status column for all 32 catalogue entries with the following script:

```python
import json
from claude_binder.paths import package_file

tools = json.loads(package_file("data", "catalog.json").read_text())["tools"]
for key, entry in sorted(tools.items()):
    print(key, entry["availability"]["status"])
```

The tables below group these tools by stage to help you choose a stack. Four of their columns answer the tool-mix decision.

**Evidence** states what the package holds for a row, strongest value first. `has run` marks an entry `catalog.json` records as `deployed`, which the status list above defines as an adapter contract that has run. `adapter ships` marks a tool that a shipped profile binds to an adapter module the package carries, with no completed run behind it. `contract only` marks a profile row whose adapter names no module. `no binding` marks a tool the package neither binds nor carries. The value reports a binding recorded in `catalog.json`, not proof that the bound adapter executes that tool, so read the Key constraint cell beside it before selecting a row. Reproduce the column for all 32 catalogue entries with this command:

```bash
PYTHONPATH="$BINDER_SKILL" python3 -B -c 'import json;from claude_binder import lane,reproduction_readiness as R;from claude_binder.paths import package_root;K=package_root();A={};[A.setdefault(a["adapter_id"],set()).add(R._module_name(a)) for f in sorted((K/"data/templates/profiles").glob("*.json")) for a in lane.load_profile(f).get("adapters") or []];C=json.loads((K/"data/catalog.json").read_text());I=lambda e:[s.strip() for s in str((e.get("profile_selection") or {}).get("adapter_id") or "").split("/")];[print(k,"|","has run" if e["availability"]["status"]=="deployed" else "adapter ships" if any(m and (K/"adapters"/(m+".py")).is_file() for i in I(e) for m in A.get(i,())) else "contract only" if any(i in A for i in I(e)) else "no binding") for k,e in sorted(C["tools"].items())]'
```

**Route** names the delivery a row is available through, using the vocabulary of [route shapes](#route-shapes) above: `platform skill` for a Claude Science runbook, `hosted API` for a provider service you call with your own credentials, `self-hosted` for your own Modal, RunPod, or Lambda account, `local` for an install on the session machine, and `in the package` for an implementation this package already carries. A row can name more than one. `unknown` means no shipped file answers it. A tool with a public licence and public weights is self-hostable whether or not an adapter for it ships here, so `unbound` in the Status column and `self-hosted` in this column are frequently both true of the same row. Read Status for what Binder can dispatch and Route for what you can run. The [NVIDIA BioNeMo NIM](#nvidia-bionemo-nim) route above is a further way to run the tools that section names, under its own contract.

**Roster** marks the tools the released campaign names for that stage, as recorded in [published campaign comparison](published-campaign-comparison.md). Choosing a `no` row departs from the reproduced stack.

**Price** links the tool's own rows in [measured costs](measured-costs.md). `none` means no run of that tool is priced there.

A `†` on a **Code licence** cell marks a tool the shipped licence gate refuses once the campaign answers `declared_use: commercial`, because `catalog.json` leaves that tool's code or weights commercial-use evidence incomplete. The gate refuses six of the 32 catalogue entries on that ground, and the licence text beside the mark still describes the tool. The marked rows are `diffdock`, `esmfold2-native-design`, `ncbi-sequence-fetch`, `protenix-v2`, `pxdesign`, `rfdiffusion3`. `boltzgen` left the list on 2026-09-13, when its code licence was read from the pinned 0.3.2 wheel and its weights licence from the model card the pinned CLI names. `proteina-complexa` and `freebindcraft` left it the same day: NVIDIA's weights licence states that its models are commercially usable, and the AlphaFold parameter archive FreeBindCraft's installer fetches is licensed CC BY 4.0 by its publisher. `pxdesign` and `proteina-complexa` joined the catalogue on 2026-09-12 when their adapters were ported, and both arrive daggered because their weights terms are unsettled. `‡` marks UCSF ChimeraX, which the gate refuses until the configuration records a separately executed written agreement reference and then admits with a scope warning. A row whose Status reads `no entry` is refused under either declared use, because the gate requires a catalogue entry. [Licence and commercial use](licence-and-commercial-use.md) carries the full report. Reproduce both marks with this command:

```bash
PYTHONPATH="$BINDER_SKILL" python3 -B -c 'import json;from claude_binder import gate;from claude_binder.paths import package_root;C=json.loads((package_root()/"data/catalog.json").read_text());[print(k,"|",",".join(sorted({p["code"] for p in gate.evaluate({"declared_use":"commercial","cofold":{"predictors":[{"id":k}]}},C)["problems"]})) or "clear") for k in sorted(C["tools"])]'
```

### Generators and co-design

| Tool | Status | Evidence | Route | Roster | Price | Backbone atoms produced | Backbone atoms required | Code licence | Hardware | Key constraint |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| RFdiffusion | operator_configured | adapter ships | self-hosted | yes | none | unknown | unknown | BSD-3-Clause | GPU (16 GB, A100) | Shared adapter ID requires an explicit checkpoint identity; the gate clears commercial use, at the medium confidence [licence and commercial use](licence-and-commercial-use.md) records, because the licence review reads no express commercial sentence |
| RFdiffusion3 | deployed | has run | hosted API | yes | [backbone generation](measured-costs.md#backbone-generation) | `N`, `CA`, `C`, `O` | unknown | BSD-3-Clause, commercial terms unresolved † | GPU (80 GB, H100) | Measured route exists; endpoint authorization and code/weights terms still gate a new run |
| Genie3 | operator_configured | adapter ships | self-hosted | yes | [backbone generation](measured-costs.md#backbone-generation) | `CA` | unknown | Apache-2.0 | GPU (24 GB, A100) | The adapter module, a pinned Modal recipe, both revisions, an environment identity, and a C-alpha designer route ship. The argv was filled in on 2026-09-12 from the shipped Modal recipe and the adapter's own argparse, which moved this row off `contract only`. Nothing has run, the arm is unpriced, and the environment `spec_sha` and image digest stay operator values |
| ESMFold2-Native-Design | operator_configured | adapter ships | platform skill, self-hosted | no | none | unknown | None | MIT † | GPU | Adapter and profiles ship; pinned runtime values and validation remain operator work; site aiming is implemented and unmeasured |
| BoltzGen | operator_configured | adapter ships | hosted API | yes | none | unknown | unknown | MIT | GPU (80 GB, H100) | The adapter module and a bound profile row ship. No upstream commit is pinned anywhere this package can read, so the source revision joins the endpoint, the model revision and the environment identity as operator values. The licence is settled against the published 0.3.2 wheel and its weights model card rather than against a commit. The served contract runs the peptide-anything protocol, caps one request at 8 designs and exposes no seed, and nothing has run under this package's contract |
| PXDesign | operator_configured | adapter ships | hosted API | yes | none | unknown | unknown | Apache-2.0, weights terms unresolved † | GPU (80 GB, H100) | The adapter module and a bound profile row ship. The endpoint, the model revision and the environment identity stay operator values, nothing has run under this package's contract, and the binder chain arrives with placeholder residue names that need downstream sequence design |
| Proteina-Complexa | operator_configured | adapter ships | hosted API | yes | none | unknown | unknown | Apache-2.0 | GPU (80 GB, H100) | The adapter module and a bound profile row ship. The endpoint, the model revision and the environment identity stay operator values, and nothing has run under this package's contract. The served contract takes one normalized chain A and returns a fixed 64-residue binder, and the two checkpoints carry the NVIDIA Open Model License |
| FreeBindCraft | operator_configured | adapter ships | hosted API | yes | none | unknown | unknown | MIT, weights terms unresolved | GPU (80 GB, H100) | The adapter module and a bound profile row ship. The endpoint, the model revision and the environment identity stay operator values, and nothing has run under this package's contract. The arm publishes a staging set rather than accepted binders, because four of the tool's default filter thresholds compare a constant behind `--no-pyrosetta`. The application reserves chain B for the binder and refuses a target on that letter |
| BindCraft | no entry | no binding | self-hosted | no | none | full | unknown | MIT (non-commercial dependencies) | GPU | Filters require PyRosetta or `--stage-all`. Separated from the FreeBindCraft row on 2026-09-12, when the fork was catalogued and this original was not |
| BindCraft2 | operator_configured | adapter ships | local process in user-controlled compute | no | [Modal timing; dollars pending](current-status.md#bindcraft2-on-modal-2026-09-21) | unknown | unknown | LicenseRef-BindCraft2-Source-Available-Hosting-Restricted | GPU (A100-80GB run recorded; peak memory unmeasured) | The process adapter, bound profile, and Modal image recipe ship. The 2026-09-21 Modal run verified toolcheck, upstream execution, and parsing. Full Binder execution and downstream screening remain unverified. Claude Science resolves executable, model, environment, and storage values. Private cloud use is permitted under the user-installed or sole-benefit terms; third-party hosted provision has separate restrictions. See the [setup details](tool-catalogue-details.md#bindcraft2-bound-on-2026-09-21). |

One row in that table carries `no entry`. BindCraft is recorded under [tools you bring up yourself](#tools-you-bring-up-yourself), and no catalogue status can describe it, because `catalog.json` holds no entry to carry one. FreeBindCraft and BoltzGen both left that list on 2026-09-12, and each now reads `operator_configured` against a bound profile row. PXDesign and Proteina-Complexa left that list on 2026-09-12, and [two hosted generators bound on 2026-09-12](#two-hosted-generators-bound-on-2026-09-12) gives the pins each one served with.

### Sequence designers

| Tool | Status | Evidence | Route | Roster | Price | Backbone atoms produced | Backbone atoms required | Code licence | Hardware | Key constraint |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ProteinMPNN | deployed | has run | platform skill, hosted API, self-hosted | no | [sequence design](measured-costs.md#sequence-design) | None (copies input) | `N`, `CA`, `C`, `O` | MIT | GPU (8 GB) or CPU | Measured canary exists; each new run still supplies its checkout, weights, and route |
| SolubleMPNN | operator_configured | adapter ships | platform skill, self-hosted | yes | [sequence design](measured-costs.md#sequence-design), cost unrecorded | None (copies input) | `N`, `CA`, `C`, `O` | MIT | GPU (8 GB) or CPU | Shared adapter ships; qualified soluble checkpoint runtime remains operator work |
| LigandMPNN | operator_configured | adapter ships | self-hosted | no | none | None (copies input) | `N`, `CA`, `C`, `O` | MIT for code and model parameters | CPU, GPU when present | Bring your own checkout and checkpoint through `--ligandmpnn-root`; `ligandmpnn-designer.template.json` swaps it in for ProteinMPNN |
| SolubleCaliby | operator_configured | adapter ships | local | yes | none | unknown | unknown | Apache-2.0 | CPU, GPU when present | The released protocol runs it in fixed-backbone mode on a 32-structure Protpardelle-1c ensemble. The fixed-backbone adapter and its `rfdiffusion3-two-arm.template.json` binding shipped on 2026-09-13. The operator supplies the pinned checkout, interpreter, checkpoint and its SHA-256. No completed run records a qualification receipt, cost or hardware measurement, and the separate ensemble mode is unqualified |

### Co-folding predictors

| Tool | Lineage / family | Status | Evidence | Route | Roster | Price | Code licence | Hardware | Key capabilities |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ESMFold2-Full | ESM embedding | deployed | has run | platform skill, self-hosted | yes | [co-folding and screening](measured-costs.md#co-folding-and-screening) | Apache-2.0 skill, MIT weights | GPU | Single-sequence co-folding; no MSA required |
| ESMFold2-Fast | ESM embedding | deployed | has run | platform skill, hosted API, self-hosted | yes | [co-folding and screening](measured-costs.md#co-folding-and-screening), [per-request model load](measured-costs.md#fal-esmfold2-fast-the-per-request-model-load-is-the-cost) | Apache-2.0 skill, MIT weights | GPU | ESMC-6B backbone; faster screen execution |
| Boltz Cloud | Independent (Boltz-2) | operator_configured | adapter ships | hosted API | no | [co-folding and screening](measured-costs.md#co-folding-and-screening) | MIT | Provider-managed | Adapter and profile ship; `boltz-api` identity, account authorization, current price evidence, and N=1 qualification remain route work |
| Boltz-2 (local) | Independent (Boltz-2) | operator_configured | adapter ships | self-hosted | no | none | MIT | GPU | Runs the open-source `boltz` console script on an environment you build; needs an operator-recorded checkpoint SHA-256, and shares the `boltz-2` lineage with the Cloud row |
| Chai-1 | Independent | unbound | no binding | platform skill, self-hosted | no | none | Apache-2.0 | GPU | Independent consensus model. `chai_lab` installs from PyPI, so you can run it yourself; a Binder plan needs an adapter |
| OpenFold3 | Independent (AF3) | unbound | no binding | platform skill, self-hosted | no | none | Apache-2.0 | GPU | AF3-faithful co-folding. Weights are public at `OpenFold/OpenFold3` behind free gating, and `bionemo-ir` runs it, so you can run it yourself; a Binder plan needs an adapter |
| AlphaFold2-Multimer-v3 | AF2 | operator_configured | adapter ships | platform skill, hosted API | no | none | Apache-2.0 AND MIT | GPU | Adapter and profile ship; endpoint, immutable pins, and qualified image remain operator work |
| Protenix v2 | Independent | operator_configured | adapter ships | self-hosted | yes | [co-folding and screening](measured-costs.md#co-folding-and-screening) | Apache-2.0 † | GPU (H100) | Adapter and profile ship; the official v2 checkpoint answers 403 while the other eighteen pinned objects fetch, and [Protenix v2 bring-up](protenix-v2-bringup.md) sets out three checkpoint routes with what each supports; qualified CUDA image, pins and validation remain operator work |

### Interface scoring, screening, and utilities

| Tool | Role | Status | Evidence | Route | Roster | Price | Code licence | Hardware | Key constraint |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ipSAE | Interface scoring | deployed | has run | in the package | yes | none | MIT | CPU | Packaged implementation and measured calibration records ship; external reference-script pin remains provenance work. BindCraft2's own `i_pDAE` is the same construction with the pair mask and the direction reduction swapped, so `ipsae-min` is not a construction-independent screen on that generator's rows. See [a generator's own ranking score](cofold-scoring-hazards.md#a-generators-own-ranking-score-can-be-the-screening-metric-under-a-different-pair-mask) |
| DockQ | Pose scoring | deployed | has run | in the package | yes | none | MIT | CPU | Packaged sc_DockQ implementation has measured records; atom-set and multimer limitations remain disclosed |
| Fair-ESM2 | Likelihood screening | unbound | no binding | platform skill, self-hosted | no | none | MIT | GPU | Masked-LM scoring and embeddings. `fair-esm` installs from PyPI under MIT, so you can run it yourself; a Binder plan needs an adapter |
| NCBI sequence fetch | Target preparation | unbound | no binding | unknown | no | none | Platform † | CPU | Accession and locus lookups; needs adapter |
| DiffDock-L | Small-molecule docking | unbound | no binding | platform skill | no | none | MIT † | GPU | Geometry prediction only; needs adapter |
| MMseqs2 | Novelty search | operator_configured | adapter ships | local | no | none | MIT | CPU | Install, version, and local database identity remain operator work. The two adapters bound here do not execute MMseqs2: `novelty_filter` reduces precomputed metrics and `target_msa_builder` refuses the local route for want of a database and an implementation |
| UniRef90 | Reference database | operator_configured | adapter ships | local | no | none | CC-BY-4.0 | Storage | Choose, fetch, hash, and record a specific release. The bound `novelty_filter` names UniRef90 in guidance and reads no release |
| ColabFold MSA server | Target MSA generation | operator_configured | adapter ships | hosted API | no | none | MIT / CC-BY-4.0 | Network | Route ships; explicit disclosure consent, egress, and connectivity check remain required |
| PyMOL (open-source) | Structure visualization | conditional_local_install | adapter ships | local | no | none | BSD-3-Clause-like | Local | Generates scripts; requires local installation |
| UCSF ChimeraX | Structure visualization | conditional_local_install | adapter ships | local | no | none | Non-commercial ‡ | Local | Generates scripts; ordinary licence does not authorize commercial use, and a separately executed written UCSF agreement is required for that use |
| Coordinate-projection renderer | Structure visualization | deployed | has run | in the package | no | none | MIT | CPU | Draws a C-alpha trace with the standard library, so it needs no local program, no GPU and no network; PyMOL and ChimeraX render scenes it cannot |

## Which profile to start from

Profiles use inheritance. `full-ensemble.template.json` serves as the base, and every other profile acts as an overlay that adds fields, overrides adapters, or removes them. Evaluating what a campaign executes requires inspecting both the base profile and its overlay.

Pick by claim level first. `production_scoring: true` marks profiles suitable for candidate claims.

The Generator column names the module each profile resolves, not the adapter id. Seven profiles bind the `rfdiffusion-generator` id to the RFdiffusion3 module, so read this column rather than the id. See [the profile id does not say which RFdiffusion you get](#the-profile-id-does-not-say-which-rfdiffusion-you-get).

| Profile | Base | Generator | Co-folding | Production scoring |
| --- | --- | --- | --- | --- |
| `small-run.template.json` | full-ensemble | RFdiffusion3, 10 backbones | ESMFold2-Fast | Yes |
| `small-run-modal.template.json` | small-run | RFdiffusion, 10 backbones | Inherited | No, unread Modal values |
| `supplied-candidates-modal.template.json` | small-run-modal | Complete supplied candidate records | ESMFold2-Fast | No, target-bound validation and unread Modal values |
| `small-run-optimized.template.json` | small-run | RFdiffusion3 | Inherited | Yes |
| `campaign-scale-single-arm.template.json` | small-run | RFdiffusion3 | Inherited | Yes |
| `hosted-n10.template.json` | small-run-optimized | RFdiffusion3 | ESMFold2-Fast | Yes |
| `browser-viewer.template.json` | small-run | Inherited | Inherited | Yes |
| `current-proven-stack.template.json` | full-ensemble | Inherited | Boltz | No |
| `afm-substitute-two-arm.template.json` | small-run | Inherited | ESMFold2-Fast and AlphaFold2-Multimer-v3 | No |
| `esmfold2-native-design.template.json` | small-run | ESMFold2 native design | Inherited | No |
| `ligandmpnn-designer.template.json` | small-run | Inherited | Inherited | No, the arm has never run |
| `modal-first-run-floor.json` | full-ensemble | RFdiffusion | ESMFold2-Full | No, provider canary |
| `local-contract-test.json` | full-ensemble | Fixtures | Fixtures | No, contract test |

`ligandmpnn-designer.template.json` is the worked substitution. It removes `proteinmpnn-designer` and `sequence-proteinmpnn` and puts LigandMPNN on the sequence-design decision, keeping every other stage of the small run. Supply a checkout through `--ligandmpnn-root` or `LIGANDMPNN_ROOT`. Read [connector authoring](connector-authoring.md) to write your own substitution the same way.

`small-run.template.json` provides the practical starting point. Its overlay removes five adapters declared in the base: `genie3-generator`, `proteinmpnn-designer-genie3`, `solublempnn-designer`, `esmfold2-predictor`, and `protenix-v2-predictor`. The overlay records why each removal occurred, marking dropped published arms as `not_run` with stated reasons.

The same small run has two provider routes. `small-run.template.json` reaches RFdiffusion3, ProteinMPNN, and ESMFold2-Fast over HTTPS deployments you own, and `compose` refuses it until the campaign fills the three `provider_endpoints` fields. `small-run-modal.template.json` runs the same design and scoring shape on your own Modal account through the Claude Science job surface, so it reads no endpoint fields and instead asks for a workspace, a spend ceiling, and one environment identity and image reference per adapter group. Both routes share every scientific choice, because the Modal profile inherits them. The **backbone generator** is the one difference: RFdiffusion3 has no Modal binding in this package, so the Modal route generates with RFdiffusion, and its `--contigs` value stays `__REQUIRED__` until you supply one. Fill the Modal values with the calls in [modal bring-up](modal-bring-up.md).

Two profiles change pipeline structure rather than scale. `esmfold2-native-design.template.json` removes `rfdiffusion-generator` and `proteinmpnn-designer` because native design generates backbone and sequence simultaneously. `current-proven-stack.template.json` removes ESMFold2 and Protenix predictors in favor of Boltz, which provides an out-of-family co-folder paired with an ESMFold2-family designer.

`supplied-candidates-modal.template.json` also changes the pipeline structure. It removes backbone
generation and sequence design, then binds normalization scale to a supplied candidate manifest.
Materialization copies and hashes each row's FASTA, candidate structure, and design pose before the
first paid fold.

Reproduce this table with the following script:

```python
import json
from claude_binder.paths import package_root

profile_dir = package_root() / "data" / "templates" / "profiles"
for path in sorted(profile_dir.glob("*.json")):
    profile = json.loads(path.read_text())
    overlay = profile.get("overlay")
    if not overlay:
        continue
    top = overlay.get("top_level") or {}
    print(path.name, "<-", profile.get("base_profile"),
          "removes:", overlay.get("remove_adapters"),
          "production_scoring:", (top.get("profile") or {}).get("production_scoring"))
```

## Which adapters are wired

An adapter is a JSON contract in the profile's `adapters` list. It specifies the role, artifacts, revisions, and three argv templates: `toolcheck_argv`, `command_argv_template`, and `parser_argv_template`. The executor renders those templates and runs them with `shell=False`.

A contract with `__REQUIRED__` fields defines an unfilled slot. Materialization rejects incomplete slots and names the JSON path requiring a value, preventing unexpected failures during paid stages.

In `full-ensemble.template.json`, 24 adapters are declared and 23 resolve to an existing package
module. The one that does not is `control-builder`, which names no module because its argv
templates are an unfilled slot. The count was 22 until 2026-09-12, when Genie3's argv was filled
and `genie3-generator` began resolving. Nine of the 24 carry `__REQUIRED__` in an execution field. Three of them leave a field the dry run reads:

| Adapter | State | What you must supply |
| --- | --- | --- |
| `genie3-generator` | Partial slot | The `spec_sha` in `environment_identity`, and `resources.container_image_digest`. Its three argv templates were filled on 2026-09-12, and its source and model revisions are pinned |
| `control-builder` | Slot | Environment identity, revisions, and argv templates for scrambled-sequence baselines |
| `rfdiffusion-generator` | Partial slot | The `--contigs` value in `command_argv_template`, `resources.container_image_digest`, `model_revision`, and the `spec_sha` in `environment_identity`. Its module resolves and its other two argv templates are complete |

Reproduce that adapter check with the following script:

```python
import json, importlib.util
from claude_binder.paths import package_file

profile = json.loads(package_file(
    "data", "templates", "profiles", "full-ensemble.template.json"
).read_text())
for a in profile["adapters"]:
    argv = a.get("command_argv_template") or []
    module = next((argv[i + 1] for i, t in enumerate(argv) if t == "-m" and i + 1 < len(argv)), None)
    resolved = bool(module) and importlib.util.find_spec(module) is not None
    print(a["adapter_id"], module, resolved)
```

`local-contract-test.json` and `provider-canary.template.json` use fixtures at `contract-test` and `canary` claim levels. `full-ensemble.template.json` binds `genie3-generator` in its base declaration, with its argv filled on 2026-09-12 and its environment identity still an operator value. Every profile with `production_scoring: true` removes `genie3-generator` in its overlay.

The package carries an adapter module at `claude_binder/adapters/genie3_generator.py` and a pinned Modal recipe at [genie3_generator_gpu.py](../envs/genie3_generator_gpu.py). The profile, catalogue, shape guard, and licence declarations come from files the package ships. These files describe a route; the package does not establish a completed Claude Binder Genie3 run or a price for it.

## Tools you bring up yourself

Detailed bring-up records preserve the pins, licence terms, runtime contracts, measurements, and
qualification limits for tools that need operator setup. Start with [tool bring-up](tool-bringup.md)
for a new environment. Read the linked detail before selecting a route or making a claim about
execution.

### Two hosted generators bound on 2026-09-12

[PXDesign and Proteina-Complexa](tool-catalogue-details.md#two-hosted-generators-bound-on-2026-09-12)
have shipped adapters and hosted contract-test profiles. Neither has run under this package's
contract.

### FreeBindCraft bound on 2026-09-12

[FreeBindCraft details](tool-catalogue-details.md#freebindcraft-bound-on-2026-09-12) cover its hosted
route, missing local bring-up, staging-set semantics, chain contract, and recorded measurements.

### BoltzGen bound on 2026-09-12

[BoltzGen details](tool-catalogue-details.md#boltzgen-bound-on-2026-09-12) cover its unpinned
upstream revision, request batching, native filter record, seed behavior, input digest checks, and
commercial-use evidence.

### SolubleCaliby bound on 2026-09-13

[SolubleCaliby details](tool-catalogue-details.md#solublecaliby-bound-on-2026-09-13) cover the
fixed-backbone route, pinned checkout and checkpoint inputs, and the separate unqualified ensemble
modes.

### BindCraft2 bound on 2026-09-21

[BindCraft2 details](tool-catalogue-details.md#bindcraft2-bound-on-2026-09-21) cover private compute
setup, licence terms, executable vocabulary, environment and weights, route timing, modality, scoring
comparison, target provenance, and parser receipt.

## Per-tool notes

[Per-tool integration notes](tool-catalogue-details.md#per-tool-notes) cover the following anchors:

### Genie3 wiring contract

Read the [Genie3 wiring contract](tool-catalogue-details.md#genie3-wiring-contract) for its input,
output, environment, hardware, weight, and licence requirements.

### The profile id does not say which RFdiffusion you get

Read the [RFdiffusion identity note](tool-catalogue-details.md#the-profile-id-does-not-say-which-rfdiffusion-you-get)
before reporting which model a shared adapter runs.

### A hosted generator can swap the chains

Read the [hosted chain assignment note](tool-catalogue-details.md#a-hosted-generator-can-swap-the-chains)
before publishing a hosted generator's binder chain.

### RFdiffusion3 needs strict PDB and a real ligand code

Read the [RFdiffusion3 input note](tool-catalogue-details.md#rfdiffusion3-needs-strict-pdb-and-a-real-ligand-code)
before dispatching ligand or PDB inputs.

### ESMFold2 has a package-identity trap and a size limit

Read the [ESMFold2 package and size note](tool-catalogue-details.md#esmfold2-has-a-package-identity-trap-and-a-size-limit)
before selecting a runtime or ligand input.

### Residue numbering drifts between the target and the prediction

Read the [residue numbering note](tool-catalogue-details.md#residue-numbering-drifts-between-the-target-and-the-prediction)
before comparing predicted structures.

### A single predictor's confidence is not evidence of binding

Read the [independent scoring note](tool-catalogue-details.md#a-single-predictors-confidence-is-not-evidence-of-binding)
when selecting confirmation arms and controls.

### FreeBindCraft's own filters go inert without PyRosetta

Read the [FreeBindCraft filter note](tool-catalogue-details.md#freebindcrafts-own-filters-go-inert-without-pyrosetta)
before interpreting its acceptance fields.

### Boltz and ESMFold2 write mmCIF into nested directories

Read the [output layout note](tool-catalogue-details.md#boltz-and-esmfold2-write-mmcif-into-nested-directories)
when configuring parsers or artifact collection.

### DockQ is undefined, not zero, without reference contacts

Read the [DockQ reference-contact note](tool-catalogue-details.md#dockq-is-undefined-not-zero-without-reference-contacts)
when a pose lacks reference contacts.

## Provider limits you will hit

Read [provider limits](tool-catalogue-details.md#provider-limits-you-will-hit) before setting request
deadlines, sequence widths, storage, or shard boundaries. The detail records hosted request limits,
MPNN validation counts, artifact-write timing, and Protenix cold-start cost.

## Choosing a stack

Choose among published-stack reproduction, stage substitution, and a changed pipeline shape after
comparing the route matrix, profile table, status, cost evidence, and scientific claim. Read
[stack selection and the worked plan](tool-catalogue-details.md#choosing-a-stack) for the tradeoffs,
scoring independence guidance, and the seven-step example.

### A worked plan

The [worked plan](tool-catalogue-details.md#a-worked-plan) gives a concrete route, fan-out, budget,
stopping rule, and scoring example.

## Open gaps in this catalogue

[Open gaps](tool-catalogue-details.md#open-gaps-in-this-catalogue) records the unbound rows, the
MMseqs2 and UniRef90 bring-up, the Protenix checkpoint routes, and the baseline-profile limitation.
