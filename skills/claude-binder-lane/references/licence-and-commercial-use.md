# Licence and commercial use

**Answer.** [`catalog.json`](../claude_binder/data/catalog.json) currently contains 32 tools. For a commercial campaign, the shipped gate clears 25 entries, refuses 6 entries because code or weight evidence is missing or unresolved, and treats ChimeraX conditionally: its ordinary UCSF Non-Commercial License does not authorize commercial use, while a separately executed written UCSF agreement can authorize that use.

`ChimeraX` is `permitted_with_conditions` in the catalog. The gate refuses commercial use until the executed agreement reference is recorded, then emits a scope-verification warning.

The gate's current result is authoritative for selection. The evidence table
below preserves the route review and names the source or TODO for each row; a
missing catalog field is an evidence gap, not a finding that commercial use is
legally prohibited.

This 2026 page gives engineering guidance for campaign selection, and qualified counsel should confirm the exact release, artifact, deployment method, and intended use.

## Sources and confidence

The table uses [`catalog.json`](../claude_binder/data/catalog.json) for package identities and roles.

The 2026-08-23 [`tool-licences.md`](tool-licences.md) review supplies the package's upstream licence reads.

The Claude Science `0.1.41-release` [platform snapshot](evidence/claude-science-platform-skills.md) supplies a 2026-08-29, SHA-256-addressed record for shipped tools.

The package [tool catalogue](tool-catalogue.md) supplies route-specific gaps for `RFdiffusion3` and `Genie3`.

**High** confidence means a named source such as [`tool-licences.md`](tool-licences.md) resolves both code and weights for the selected route.

**Medium** confidence marks a 2026 class mapping, a family-level inference, or a route condition that still needs release-time confirmation.

An `unknown (high)` row means the named 2026 sources verify that the deciding term is absent or unresolved.

## Mapping the 65-tool seed classes

The 65-tool seed vocabulary and the 4-value `commercial_use` enum answer different questions, so this page applies the following explicit mapping.

| Seed class | Package value | Reading or judgment |
| --- | --- | --- |
| `permissive` | `permitted` | **Judgment:** both code and weights must have a named MIT, Apache-2.0, BSD, or equivalent licence, with no added field-of-use restriction. |
| `open-restricted` | `permitted_with_conditions` or `unknown` | **Judgment:** use `permitted_with_conditions` only when every added condition is recorded and commercial-compatible; use `unknown` for an unverified condition. |
| `non-commercial` | `forbidden` | **Reading:** `non-commercial` limits use to research or academic contexts, which excludes a commercial campaign. |
| `proprietary` | `unknown` unless service terms answer the use case | **Judgment:** `proprietary` alone does not state commercial output rights for a closed or paid route. |
| `unverified` | `unknown` | **Reading:** `unverified` means no recorded licence name settles code, weights, or service terms. |

Each of the 5 mappings appears explicitly in the table.

## What the gate enforces today

This page summarizes the licence evidence; the shipped gate remains the executable decision for a selected route.

[`gate.py`](../claude_binder/gate.py) reads [`catalog.json`](../claude_binder/data/catalog.json) and refuses a commercial campaign that selects any tool whose catalog entry leaves code or weight evidence incomplete. Measured against the shipped catalog on 2026-09-21, that is 25 tools clear, 6 tools refused, and `chimerax` admitted only when a separate written agreement reference is recorded.

The 6 the gate refuses are `diffdock`, `esmfold2-native-design`, `ncbi-sequence-fetch`, `protenix-v2`, `pxdesign`, `rfdiffusion3`. `boltzgen` left the list on 2026-09-13: its code licence was read as MIT from the published 0.3.2 wheel, whose sha256 pins the artifact, and its weights licence as MIT from the ungated model card the pinned CLI names, whose declared base model carries the same term. `pxdesign` and `proteina-complexa` joined the catalogue on 2026-09-12 when their adapters were ported. PXDesign remains refused: it now has Apache-2.0 code read from its pinned LICENSE and an express commercial sentence in its README, but that grant names the project and not the diffusion checkpoint it downloads on first run. `proteina-complexa` and `freebindcraft` left the list on 2026-09-13. The NVIDIA Open Model License that covers the Proteina-Complexa checkpoints states that its models are commercially usable and that NVIDIA claims no ownership in outputs, so that row is `permitted_with_conditions` on the conditions the licence names. FreeBindCraft's weight layer is the AlphaFold 2 parameter archive its installer fetches, and the AlphaFold publisher licenses those parameters CC BY 4.0 and names that archive, so it clears the way `alphafold-multimer-v3` already does, on CC-BY-4.0 attribution and with the archive digest still unpinned. Five entries left that list on 2026-09-10, and `genie3` left it on 2026-09-11 when its weights licence was read from the model card at the weights host rather than from a GitHub README that states none. `ligandmpnn` left it when its adapter shipped and its code and weight licences resolved to MIT. `rfdiffusion`, `chai1`, `openfold3` and `fair-esm2` left it when [tool licences](tool-licences.md) gained rows quoting each publisher's own licence and README.

One row runs the other way. This page marks `colabfold-msa-server` `unknown (high)` for the service, and `catalog.json` records its code and weights as permitted and carries the service question as a gate condition, so the gate admits the tool and prints that condition as a warning.

**The gate reads the catalog's own `commercial_use` vocabulary and refuses any value outside it.** That covers both layers. A weights value of `forbidden` refuses with `WEIGHTS_COMMERCIAL_USE_FORBIDDEN`, and no agreement reference lifts it. An absent value, a hole marker, or a misspelling refuses, because a licence field that clears on an unrecognized word reads a typo as permission. `permitted_with_conditions` clears only when the entry states each condition at `gate.conditions`, which is the rule the seed-class mapping above already sets for that value; the pass then carries a `WEIGHTS_COMMERCIAL_USE_CONDITIONAL` warning naming every condition to satisfy. `alphafold-multimer-v3` is the one shipped row that takes that path, on CC-BY-4.0 attribution and the AlphaFold2 model-parameter terms. Each of those four cases cleared the gate silently until 2026-09-12.

**The gate reads the catalog's own `weights_status_enum` and refuses any status outside it.** A weights row's status records how far its reading got, and an unfinished reading is now its own finding. `unverified`, `open_todo`, and the three spellings of an unfilled field (`missing`, `MISSING`, `__REQUIRED__`) leave the artifact unsettled, so a row carrying one refuses with `WEIGHTS_PROVENANCE_UNVERIFIED` even where its licence permits the use. `WEIGHTS_COMMERCIAL_USE_UNKNOWN` now means one thing only: the catalog has not settled what the weights licence permits. A null status, an absent status, or a misspelling refuses with `WEIGHTS_STATUS_UNRECOGNIZED`. This check was a denylist of four spellings until 2026-09-12, so `unverified` and every invented word cleared it. No shipped verdict moves. At the 28-row checkpoint, running the earlier gate against that catalogue cleared and refused all 28 rows exactly as the then-current one did, and changed a single refusal reason: `protenix-v2` reports `WEIGHTS_PROVENANCE_UNVERIFIED` where it reported `WEIGHTS_COMMERCIAL_USE_UNKNOWN`, on an `open_todo` weights row whose licence reads permitted. At that checkpoint, `proteina-complexa`, `pxdesign` and `rfdiffusion3` already refused commercial selection on an unresolved weights licence. The later BoltzGen and FreeBindCraft rows carried unverified weight records at that checkpoint, and both were read on 2026-09-13.

**The gate reads the catalog's own `code_licence_status_enum` and refuses any status outside it.** A code-licence row's status records how far its reading got. `unverified`, `open_todo`, and the two spellings of an unfilled field (`MISSING`, `__REQUIRED__`) leave the licence record unread, so a row carrying one refuses with `LICENCE_PROVENANCE_UNVERIFIED` even where the `commercial_use` value beside it permits the use. An open record is not a ban, so that refusal never reads as one. A null status, an absent status, or a misspelling refuses with `LICENCE_STATUS_UNRECOGNIZED`. Nothing read `code_licence.status` until 2026-09-12, so every one of those cases cleared. At the 28-row checkpoint no shipped verdict moved: four rows recorded an open status, `pxdesign` as `unverified` and `diffdock`, `esmfold2-native-design` and `ncbi-sequence-fetch` as `open_todo`, and all four already refused on a code `commercial_use` of `unknown`. BoltzGen and FreeBindCraft subsequently joined with unverified code records.

**A catalog declaration narrows what the gate accepts and never widens it.** Each of the three vocabularies is intersected with the values `gate.py` implements a rule for, so a value the catalog declares and the gate decides nothing about is refused rather than cleared. The declarations were taken whole until 2026-09-12: adding `perrmited` to `commercial_use_enum` and writing `perrmited` into a tool's licence row cleared a commercial campaign that selected it, and the same edit against `weights_status_enum` cleared a weights row whose provenance nobody had read. `catalog.json` is the document this gate polices, so a one-line edit to it must not be able to authorize a commercial use.

A declared non-commercial campaign reaches none of this. The gate checks only that every selected tool has a catalog entry, so all 32 rows are available.

This page and `catalog.json` are not reconciled. The gate enforces evidence the package holds rather than a citation it has not opened, so a row that reads `permitted` here can still be refused.

## The 32 catalogued tools

`N/A` means the named tool has no model-weight layer in the 2026 evidence.

| Tool | What it does | Code licence | Weights licence | Commercial use and confidence | Grounding or TODO |
| --- | --- | --- | --- | --- | --- |
| `rfdiffusion` | Generates binder backbones | BSD-3-Clause | BSD-3-Clause | `permitted` (medium) | [`catalog.json`](../claude_binder/data/catalog.json) and [RFdiffusion licence review](tool-licences.md); seed `permissive` to package `permitted` is a judgment because the local review records no express commercial sentence. |
| `rfdiffusion3` | Generates binder backbones | BSD-3-Clause | `unknown` | `unknown` (high) | [Tool catalogue](tool-catalogue.md#tools-you-bring-up-yourself); **TODO:** obtain checkpoint terms from the RFdiffusion3 publisher that cover use, redistribution, and outputs. |
| `genie3` | Co-designs backbone and sequence | Apache-2.0 | Apache-2.0 | `permitted` (high) | [Licence review](tool-licences.md); the model card at the `yeqinglin/genie3` weights host declares `apache-2.0`, which settles the seed's `permissive` class against the earlier missing-weight finding. The weights revision is `9ae31ebb8c56eebdc05ab282a8fd3f6a6d2a03a2`. |
| `pxdesign` | Generates binder backbones | Apache-2.0 | `unknown` | `unknown` (high) | [`catalog.json`](../claude_binder/data/catalog.json) and [licence review](tool-licences.md); Apache-2.0 read from the LICENSE at `bytedance/PXDesign@f788441313c84c3074fe9596ac2433f96b15c763`, whose README states the project is free for both academic research and commercial use. **TODO:** obtain terms for the diffusion checkpoint and the three Protenix checkpoints the pinned CLI fetches on first run, which that grant does not name. |
| `proteina-complexa` | Co-designs a complex and its binder sequence | Apache-2.0 | NVIDIA Open Model License | `permitted_with_conditions` (high) | [`catalog.json`](../claude_binder/data/catalog.json) and the [licence review](tool-licences.md) Proteina-Complexa weights row, which quotes the NVIDIA Open Model License clauses stating that models are commercially usable and that NVIDIA claims no ownership in outputs; the conditions are licence compliance, the incorporated NVIDIA Trustworthy AI terms, and an attribution notice on redistribution. **TODO:** read `licenses/license_weights.txt` at the pinned revision and record which dated version of the agreement it carries. |
| `boltzgen` | Co-designs binder structure and sequence | MIT | MIT | `permitted` (high) | [`catalog.json`](../claude_binder/data/catalog.json) and the [licence review](tool-licences.md) BoltzGen rows; MIT read from the published `boltzgen` 0.3.2 wheel, whose sha256 pins the artifact, and MIT declared by the ungated `boltzgen/boltzgen-1` model card at revision `c1be29e1f82ffcc72264f64b993c43fb4e0d17f0`, whose declared base model carries the same term. MIT notice obligations still apply to distributed software copies. |
| `freebindcraft` | Designs structure and sequence through AlphaFold 2 | MIT | CC-BY-4.0 | `permitted_with_conditions` (high) | [`catalog.json`](../claude_binder/data/catalog.json) and the [licence review](tool-licences.md) FreeBindCraft rows; MIT read from the LICENSE at the pinned fork commit, and the AlphaFold model-parameters terms, which license the parameters CC BY 4.0 and name `alphafold_params_2022-12-06.tar`, the archive the installer fetches. CC-BY-4.0 attribution is required and the archive digest is still unpinned. |
| `bindcraft2` | Designs structure and sequence through AlphaFold 2, from a copy the user installed | LicenseRef-BindCraft2-Source-Available-Hosting-Restricted | MIT AND CC-BY-4.0 | `permitted_with_conditions` (high) | [`catalog.json`](../claude_binder/data/catalog.json) and the [licence review](tool-licences.md) BindCraft2 row. The licence at the v1.0.0 commit grants free use for any purpose including internal commercial use, names asset development and drug discovery inside that grant, and restricts provision to third parties as a hosted, managed, cloud, API or workflow-platform offering, counting an invocable tool or agent action inside a hosted platform as one. What it conditions is deployment surface and attribution rather than commercial purpose, and `catalog.json` records all five conditions. The weights layer is MIT for the twelve ProteinMPNN and HyperMPNN checkpoints the package ships and CC BY 4.0 for the AlphaFold 2 parameters it fetches on first use. HyperMPNN's own repository publishes MIT and states no weights-specific term, so the positive variant's reading rests on BindCraft2's attribution, and that variant is not the default. The parameter archive digest is still unpinned. |
| `proteinmpnn` | Designs sequences for fixed backbones | MIT | MIT | `permitted` (high) | [Pinned ProteinMPNN evidence](tool-licences.md); MIT notice obligations still apply to distributed software copies. |
| `solublempnn` | Designs soluble-biased sequences | MIT | MIT | `permitted` (high) | [Pinned SolubleMPNN evidence](tool-licences.md); the `8907e6671bfbfc92303b5f79c4b5e6ce47cdef57` source tree contains the checkpoint. |
| `solublecaliby` | Designs sequences on fixed backbones, with a separate ensemble mode | Apache-2.0 | Apache-2.0 | `permitted` (medium) | [Pinned upstream reading](evidence/solublecaliby-upstream-2026-09-13.md); verify the local checkpoint digest and qualify the selected runtime. |
| `ligandmpnn` | Designs sequences around ligands, nucleic acids, or metals | MIT | MIT | `permitted` (high) | [pinned LICENSE](https://raw.githubusercontent.com/dauparas/LigandMPNN/26ec57ac976ade5379920dbd43c7f97a91cf82de/LICENSE) and [pinned README](https://raw.githubusercontent.com/dauparas/LigandMPNN/26ec57ac976ade5379920dbd43c7f97a91cf82de/README.md), which states that the code and the model parameters are both MIT. |
| `boltz` | Predicts co-folded complexes | MIT | MIT | `permitted` (high) | [Licence review](tool-licences.md); **TODO for hosted routes:** record the selected provider's service terms. |
| `boltz-local` | Folds a binder against its target with the open-source package | MIT | MIT | `permitted` (high) | [Boltz licence review](tool-licences.md); same MIT as the hosted route, and the upstream downloader verifies no checkpoint bytes, so this arm requires an operator-recorded SHA-256. |
| `chai1` | Predicts protein, nucleic-acid, and ligand complexes | Apache-2.0 | Apache-2.0 | `permitted` (high) | [Claude Science snapshot](evidence/claude-science-platform-skills.md#chai1) and [publisher licence](https://github.com/chaidiscovery/chai-lab/blob/main/LICENSE); **TODO for `--use-msa-server`:** record ColabFold service terms. |
| `openfold3` | Predicts protein, nucleic-acid, and ligand complexes | Apache-2.0 | Apache-2.0 | `permitted` (high) for an offline run | [Claude Science snapshot](evidence/claude-science-platform-skills.md#openfold3) and [v0.5.0 licence](https://raw.githubusercontent.com/aqlaboratory/openfold-3/v0.5.0/LICENSE); **TODO for the default MSA server:** record ColabFold service terms. |
| `esmfold2` | Predicts co-folded structures | MIT | MIT | `permitted_with_conditions` (high) | [Licence review](tool-licences.md) and [Claude Science snapshot](evidence/claude-science-platform-skills.md#esmfold2); retain MIT and third-party notices, and apply the Biohub service AUP when the API route is selected. |
| `esmfold2-fast` | Runs the faster ESMFold2 predictor | MIT | MIT | `permitted_with_conditions` (high) | [Licence review](tool-licences.md); retain MIT and third-party notices, and apply Biohub service terms when the API route is selected. |
| `esmfold2-native-design` | Designs a whole-surface binder backbone and sequence | MIT | MIT | `unknown` (high) | [`catalog.json`](../claude_binder/data/catalog.json) and [Biohub ESMFold2 model card](https://huggingface.co/biohub/ESMFold2); the catalog records both commercial-use values as `unknown`, so the gate refuses this row. The platform states Apache-2.0 for the `esmfold2` skill and MIT for the ESMFold2 weights and makes no statement about the Experimental variants, so a design-specific licence source is absent and the ESMFold2 family terms are an inheritance rather than a reading. **TODO:** read the `Biohub/esm` LICENSE covering the `biohub/ESMFold2-Experimental*` repositories. |
| `alphafold-multimer-v3` | Predicts multimer structures | Apache-2.0 AND MIT | CC-BY-4.0 | `permitted_with_conditions` (high) | [`catalog.json`](../claude_binder/data/catalog.json) and [AlphaFold parameter terms](https://github.com/google-deepmind/alphafold#model-parameters-license); CC-BY-4.0 attribution is required. |
| `protenix-v2` | Predicts co-folded complexes | Apache-2.0 | Apache-2.0 | `permitted` (high) | [Protenix licence review](tool-licences.md) and [publisher README](https://raw.githubusercontent.com/bytedance/Protenix/main/README.md), which cover code and model parameters. |
| `ipsae` | Scores predicted interfaces | MIT | N/A | `permitted` (high) | [ipSAE licence review](tool-licences.md). |
| `dockq` | Scores pose and interface quality | MIT | N/A | `permitted` (high) | [DockQ licence review](tool-licences.md). |
| `fair-esm2` | Scores sequence likelihood and mutation effects | MIT | MIT | `permitted` (high) | [Claude Science snapshot](evidence/claude-science-platform-skills.md#fair-esm2) and [Meta ESM licence](https://raw.githubusercontent.com/facebookresearch/esm/main/LICENSE); the platform skill wrapper's Apache-2.0 licence is separate from ESM-2's MIT terms. |
| `ncbi-sequence-fetch` | Fetches target sequences and records | `unknown` | N/A | `unknown` (high) | [`catalog.json`](../claude_binder/data/catalog.json) records this as an optional, unbound retrieval route. Check the selected implementation's code licence and the reuse terms for each retrieved NCBI record type. |
| `diffdock` | Generates and ranks small-molecule poses | MIT on unpinned `main` | MIT on unpinned `main` | `unknown` (medium) | [Claude Science snapshot](evidence/claude-science-platform-skills.md#diffdock); **TODO:** pin the selected DiffDock commit and verify the code licence plus every checkpoint term at that pin. |
| `mmseqs2` | Searches sequences and novelty databases | MIT | N/A | `permitted` (high) | [MMseqs2 licence review](tool-licences.md); the 2026 campaign must also pin each searched database release and its terms. |
| `uniref90` | Supplies a clustered protein-sequence reference database | CC-BY-4.0 | N/A | `permitted_with_conditions` (high) | [UniRef90 licence review](tool-licences.md); CC-BY-4.0 attribution is required. |
| `colabfold-msa-server` | Supplies remote MMseqs2 alignments | MIT client code | CC-BY-4.0 AlphaFold parameters | `unknown` (high) for the service | [ColabFold licence review](tool-licences.md); **TODO:** obtain published `api.colabfold.com` service and sequence-use terms before sending a commercial target. |
| `pymol-open-source` | Renders and inspects structures | PyMOL open-source BSD-like licence | N/A | `permitted_with_conditions` (high) | [PyMOL licence review](tool-licences.md); preserve notices and follow trademark conditions. |
| `chimerax` | Renders and analyses structures | UCSF ChimeraX Non-Commercial License Agreement | N/A | `permitted_with_conditions` (high) | [UCSF licence review](tool-licences.md); the ordinary licence does not authorize commercial use, and a separately executed written UCSF agreement can authorize the route. |
| `python-coordinate-projection` | Renders a backbone picture with the standard library | MIT | N/A | `permitted` (high) | The package's own [MIT licence](../LICENSE.txt); the renderer ships inside the package, so it adds no third-party term. |

## Weights, code, and outputs

The 32-tool decision separates code from weights because [`catalog.json`](../claude_binder/data/catalog.json) stores the 2 layers independently.

`RFdiffusion3` pairs permissive code with unverified weights, so its output obligations remain `unknown`. `Genie3` no longer does: the weights host declares `apache-2.0` for the checkpoint.

`DiffDock-L` has an MIT statement on unpinned `main`, while the selected checkpoint and source revision remain unpinned in [`catalog.json`](../claude_binder/data/catalog.json).

A 2026 commercial campaign must treat unverified weight terms as unresolved for checkpoint use, redistribution, and generated outputs.

No 2026 source reviewed here grants free output rights by silence.

`ProteinMPNN`, `Chai-1`, `OpenFold3`, and `Protenix v2` reach a grounded answer because named sources cover both code and weights.

## Non-commercial tools outside the 32-tool catalogue

The 65-tool seed marks 4 tools outside [`catalog.json`](../claude_binder/data/catalog.json) as `non-commercial`, which maps to `forbidden` for a commercial 2026 campaign.

| Tool | Why the 2026 route is non-commercial | Commercial action | Primary grounding |
| --- | --- | --- | --- |
| `BindCraft` | The `BindCraft` installer adds PyRosetta and AlphaFold2 parameters. | `BindCraft` needs separate commercial terms for every component before a commercial run. | [BindCraft installer](https://raw.githubusercontent.com/martinpacesa/BindCraft/main/install_bindcraft.sh) and [Rosetta commercial licensing](https://www.rosettacommons.org/software/licensing-and-tutorials). |
| `Germinal` | The `Germinal` run requires an academic PyRosetta licence. | `Germinal` needs a separate PyRosetta commercial licence before a commercial run. | [Germinal README](https://raw.githubusercontent.com/SantiagoMille/germinal/main/README.md) and [Rosetta academic terms](https://downloads.rosettacommons.org/software/academic/). |
| `alphafold3` original parameters | Apache-2.0 covers AlphaFold 3 code, while the AlphaFold 3 Model Parameters Terms govern weights. | `alphafold3` needs an approved commercial parameter route or another predictor. | [AlphaFold 3 README](https://raw.githubusercontent.com/google-deepmind/alphafold3/main/README.md). |
| `pyrosetta` | The Rosetta Software Non-Commercial License Agreement covers the PyRosetta academic download. | `pyrosetta` needs a Rosetta commercial licence before commercial analysis. | [Rosetta academic terms](https://downloads.rosettacommons.org/software/academic/). |

Each of the 4 routes needs written commercial terms before a 2026 campaign spends compute or relies on outputs.

## The 7 placeholders

`ligandmpnn` is closed: its row above carries MIT for the code and the model parameters, read from the pinned upstream `LICENSE` and `README`. Three of the remaining 6 have closed from the 2026 evidence: `chai1`, `fair-esm2`, and `openfold3`.

Three of the 7 named placeholder tools stay open: `diffdock`, `esmfold2-native-design`, and `ncbi-sequence-fetch`.

`ncbi-sequence-fetch` needs a shipped-code licence and record-type reuse terms.

`diffdock` needs a selected source commit plus checkpoint terms at that commit, because an unpinned `main` read does not settle a future campaign pin.

`esmfold2-native-design` needs a licence source for the ESMFold2-Experimental variants, which the 0.1.41 platform snapshot does not cover.
