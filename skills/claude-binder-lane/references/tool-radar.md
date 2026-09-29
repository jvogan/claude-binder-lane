# Tool radar

Use this page to widen the option set before selecting a stack. It records
capability leads. It does not claim that a tool is installed, served, licensed
for the campaign, scientifically suitable, or bound to Binder.

Start with `binder_cli(["tools", "--json"])` for the packaged 32-tool catalogue,
then follow [Discover tools and routes](platform-tool-discovery.md) for the live
inventory. Investigate only the leads that fit the campaign question.

## Generation and co-design

| Family | Leads | Why investigate |
| --- | --- | --- |
| General backbone generation | RFdiffusion, RFdiffusion3, Genie3 | Motif scaffolding, binder backbones, or alternative diffusion families |
| End-to-end binder design | BindCraft, FreeBindCraft, BoltzGen, PXDesign, Proteina-Complexa | Joint or staged sequence and structure design with different control surfaces |
| Emerging generators | FoldCraft, ODesign, ProtPardelle-1C, AlphaProteo, Mosaic, HalluDesign | Alternative sampling or design objectives when a campaign accepts bring-up work |
| Antibody-focused generation | RFantibody | Antibody scaffold and loop constraints rather than a generic binder shape |
| Switch design | SwitchCraft | Conditional or state-dependent protein design questions |

Confirm target scope, ligand and nucleic-acid support, motif and hotspot syntax,
backbone atom set, chain order, residue numbering, binder-length behavior, seed
control, weights terms, and output parser before choosing a generator.

## Sequence design

| Family | Leads | Why investigate |
| --- | --- | --- |
| General inverse folding | ProteinMPNN, ESM inverse-folding methods | Sequence recovery and redesign from a complete backbone |
| Solubility-aware design | SolubleMPNN, Caliby, SolubleCaliby | Expression-oriented sequence priors or reranking |
| Context-aware design | LigandMPNN | Ligand, nucleic-acid, and metal context |
| Backbone-shape variants | ProteinMPNN CA-only | C-alpha-only generator outputs that a full-backbone parser misreads |
| Antibody sequence design | AbMPNN, AntiFold | Antibody framework and CDR-aware sequence choices |
| Alternative inverse folders | FAMPNN | A distinct model family for comparison or substitution |

Read multi-FASTA semantics carefully. Some outputs include a native or wild-type
row alongside designs. Count and score only the rows the route contract marks as
generated.

## Structure prediction and cross-checking

| Family | Leads | Why investigate |
| --- | --- | --- |
| AF2 lineage | AlphaFold2-Multimer, ColabFold | Published comparison, MSA-conditioned multimer prediction |
| AF3-class co-folders | OpenFold3, Protenix, RoseTTAFold3 | Protein, nucleic-acid, ligand, ion, or modified-residue complexes |
| Independent co-folders | Boltz, Chai-1 | Cross-model survivor checks and complex confidence outputs |
| Single-sequence models | ESMFold2 Full and Fast | MSA-free screens and a different embedding lineage |
| Diagnostic variants | AF-Unmasked | Research questions about masking or model behavior |

Keep local open-source, managed endpoint, hosted API, and self-hosted NIM or
container routes separate even when they expose the same model family. Their
payloads, limits, costs, retention, and version identities differ.

## Scoring, controls, search, and structure utilities

| Role | Leads | Contract to settle |
| --- | --- | --- |
| Interface confidence | ipSAE, model-specific PAE and interface confidence | Direction, aggregation, cutoff, and calibration target |
| Pose comparison | DockQ, scDockQ | Reference structure, chain mapping, atom set, and undefined cases |
| Structural similarity | TM-align, US-align | Executable pin, chain policy, normalization, and reference set |
| Secondary structure | DSSP | Executable pin and residue mapping |
| Sequence novelty | MMseqs2, ESM-2 embeddings | Database release, search parameters, offline or service route, and threshold basis |
| Structural novelty | Foldseek | Database release, query representation, and score interpretation |
| Liability and developability | OpenDDE and campaign-specific filters | Molecular scope, threshold source, and whether the output is a filter or ranking feature |
| Negative controls | Scrambled sequences, site-shifted controls, unrelated proteins | Generation rule, seed, target scope, and expected failure behavior |

Confidence from one predictor does not establish binding. Use controls and an
independent model lineage when the claim requires them. Record undefined metrics
as null with a reason instead of converting them to zero.

## Docking and chemistry-adjacent routes

DiffDock, FlowDock, NeuralPLexer, and smina are useful leads when a binder
workflow includes ligands, cofactors, competitive pockets, or geometry checks.
They answer different questions from protein-binder generation. Record molecular
protonation, stereochemistry, CCD or small-molecule identity, receptor state,
search box, pose count, and scoring meaning in their own route contract.

## Compute and service routes

Any selected tool may run locally, in a Claude Science compute environment, at
a managed endpoint, through a model API, in a user-owned container on Modal,
RunPod, Lambda, or another GPU cloud, or through SSH to a workstation, scheduler,
HPC system, or neocloud. Provider choice follows the tool's software and hardware
needs, data policy, price, queue, region, persistence, and operator preference.

Do not require one provider for the whole campaign. Normalize every stage to
hashed artifacts and receipts, then mix routes where that improves cost,
availability, privacy, or scientific independence.

## Promotion rule

A radar lead becomes a Binder route only after its model-specific request,
response, artifact, residue, authorization, cost, retry, and receipt contracts
exist. External served evidence remains useful provenance. It never satisfies a
Binder preflight by itself and never blocks a campaign that did not select it.
