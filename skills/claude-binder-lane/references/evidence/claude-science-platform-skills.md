# Claude Science 0.1.41-release, shipped skills

This file is generated. It records what the installed Claude Science runtime states about each skill it ships, so that a claim in `claude_binder/data/catalog.json` can be checked against a file that ships with the skill.

**Platform version.** `0.1.41-release`

**Read on.** 2026-08-29

**Read from.** `~/.claude-science/runtime/0.1.41-release/skills/`

**Contents.** 29 skill directories and 81 regular files totalling 883,245 bytes.

**Scoped for the installed skill.** This copy carries the 22 of 29 skill sections that fall inside computational structural biology, along with the compute and endpoint sections a campaign runs through. The other 7 are removed, with their coverage rows and their file-inventory rows, so a name that appears in another section's verbatim description may have no section here. The inventory hash below still covers the whole install, and it never covered this file.

## Why this file exists

The package runs inside Claude Science, so the platform's shipped skills are a declared host dependency. The runtime is separate from this package and a reader may not have it installed. Citing a path inside the install would therefore be a citation most readers cannot open. This snapshot is the openable form of that evidence.

The snapshot is the stated fallback for an absent platform. It carries the version and hash it was read from, so a reader can tell it apart from the live install. It records one past read and does not track later changes.

## How to cite this file

Cite a heading anchor, never a line number. Every skill below is an `h2` whose text is the skill's own `name`, so the anchor is the skill name: `references/evidence/claude-science-platform-skills.md#openfold3`.

Line numbers are wrong targets here because this file is regenerated. Any change in the install shifts every line below it, so a line-numbered citation silently starts pointing at the wrong skill. An anchor stays valid across a regeneration while the platform keeps the skill's name.

A skill that the platform drops loses its anchor, and every citation to it then fails to resolve. That failure is the point. A line-numbered citation would instead resolve to a different skill.

## What the hash covers

**Inventory SHA-256.** `d8cec5332830a8d56815ef4226d8ad840c9b15f706b5b050d999f44f39d67e6a`

That hash covers the bytes of all 81 regular files under `~/.claude-science/runtime/0.1.41-release/skills/`, including `THIRD_PARTY_LICENSES.md` and every non-Markdown file such as `provider.py` and `requirements.lock`. Symbolic links and directories are excluded. Each file is addressed by its path relative to that root.

It is the SHA-256 of the concatenation of one line per file, `"<sha256>  <relative path>\n"`, sorted by path. That is the byte stream `sha256sum` emits, so a reader can recompute it with shell tools alone:

```
cd ~/.claude-science/runtime/0.1.41-release/skills && \
  find . -type f -not -type l | sed 's|^\./||' | LC_ALL=C sort | \
  xargs shasum -a 256 | sed 's|  \./|  |' | shasum -a 256
```

The hash covers the install. It does not cover this file. A mismatch means the install changed after this snapshot was read, so every claim below needs rechecking. The hash does not say which skill changed. The per-skill `SKILL.md` SHA-256 in each section below narrows it to one directory.

## Regenerating

This snapshot covers `0.1.41-release` alone. Restamping it against another version would carry claims that were never checked against that version, so a different platform release needs a snapshot of its own.

## How to read these fields

Every value below is transcribed from the install. Where the platform states nothing, the field reads `not stated` and the section names the file that was checked. No gap is filled with a plausible value.

**Licence** is what the platform declares, for the skill itself and for the third-party code, weights, and services it names. No row here is a read of an upstream LICENSE file, and none is legal advice. `references/tool-licences.md` holds the upstream reads.

**Compute route** lists two declarations found on disk: a `requirements` key in the frontmatter, and whether the directory ships a `provider.json`. The platform has no field stating local or remote execution, so this snapshot infers nothing beyond those two.

**Description** is the skill's own `description`, reproduced verbatim with the block indent stripped. Its wording, punctuation, and line breaks are the platform's.

## Coverage

| Skill | Display name | Category | Skill licence | Third-party entries |
| --- | --- | --- | --- | --- |
| [`alphafold2`](#alphafold2) | AlphaFold2 | biomodels | Apache-2.0 | 2 |
| [`boltz`](#boltz) | not stated | biomodels | Apache-2.0 | 2 |
| [`borzoi`](#borzoi) | not stated | biomodels | Apache-2.0 | 1 |
| [`chai1`](#chai1) | Chai-1 | biomodels | Apache-2.0 | 2 |
| [`compute-env-setup`](#compute-env-setup) | not stated | not stated | Apache-2.0 | none |
| [`diffdock`](#diffdock) | DiffDock | biomodels | Apache-2.0 | 1 |
| [`esmfold2`](#esmfold2) | ESMFold2 | biomodels | Apache-2.0 | 1 |
| [`evo2`](#evo2) | Evo 2 | biomodels | Apache-2.0 | 1 |
| [`fair-esm2`](#fair-esm2) | ESM-2 | biomodels | Apache-2.0 | 1 |
| [`figure-composer`](#figure-composer) | not stated | not stated | Apache-2.0 | none |
| [`figure-style`](#figure-style) | not stated | not stated | Apache-2.0 | none |
| [`ligandmpnn`](#ligandmpnn) | LigandMPNN | biomodels | Apache-2.0 | 1 |
| [`managed-model-endpoints`](#managed-model-endpoints) | not stated | not stated | Apache-2.0 | none |
| [`openfold3`](#openfold3) | OpenFold3 | biomodels | Apache-2.0 | 2 |
| [`proteinmpnn`](#proteinmpnn) | ProteinMPNN | biomodels | Apache-2.0 | 1 |
| [`remote-compute-modal`](#remote-compute-modal) | not stated | not stated | Apache-2.0 | none |
| [`remote-compute-ssh`](#remote-compute-ssh) | not stated | not stated | Apache-2.0 | none |
| [`scgpt`](#scgpt) | scGPT | biomodels | Apache-2.0 | 1 |
| [`scvi-tools`](#scvi-tools) | scvi-tools | not stated | Apache-2.0 | none |
| [`self-awareness`](#self-awareness) | not stated | not stated | Apache-2.0 | none |
| [`solublempnn`](#solublempnn) | SolubleMPNN | biomodels | Apache-2.0 | 1 |
| [`using-model-endpoint`](#using-model-endpoint) | not stated | not stated | Apache-2.0 | none |

`THIRD_PARTY_LICENSES.md` ships beside these directories at the skills root. It is a single attribution document for the whole distribution and is not a skill, so it has no section below. Its bytes are inside the inventory hash.

## alphafold2

| Field | Value |
| --- | --- |
| Directory | `0.1.41-release/skills/alphafold2/` |
| Declared name | `alphafold2` |
| Display name | `AlphaFold2` |
| Category | `biomodels` |
| Skill licence, as the platform states it | `Apache-2.0` |
| Declared requirements | `[gpu]` |
| Compute route declarations | frontmatter `requirements: [gpu]` |
| Version or pin | not stated. The frontmatter of `alphafold2/SKILL.md` carries no version, release, or commit field. |
| Files shipped | `SKILL.md` |
| `SKILL.md` SHA-256 | `562520a36043c09f6e740b28a097e1e11a68b73bcc3c4c7cb0e269a644b4427a` |

Description, verbatim from `alphafold2/SKILL.md` frontmatter:

```
Predict protein structure for monomers and multimers with AlphaFold2 via the
ColabFold runner (Mirdita et al. 2022, github.com/sokrypton/ColabFold;
AlphaFold2 Jumper et al. 2021). Reach for this skill to fold a sequence or
complex with the AF2/AF2-Multimer evoformer, to validate designed sequences
by self-consistency pLDDT, ipTM, and RMSD, or to run a quick MSA-backed
prediction using the public MMseqs2 server.
```

Opening paragraph, verbatim from the `alphafold2/SKILL.md` body. This is where the skill names the tool and model it wraps:

```
This skill wraps AlphaFold2 and AlphaFold2-Multimer through `colabfold_batch`,
which replaces DeepMind's local-database MSA pipeline with a call to the public
MMseqs2 server — so a prediction is one command and one FASTA, not a 2 TB
database mount. AF2 remains the reference monomer predictor and the multimer
model is still a strong protein–protein validator, but it does not handle
ligands or nucleic acids; for those, route to `boltz`, `chai1`, or `openfold3`.
The ColabFold code is MIT (github.com/sokrypton/ColabFold) and the AlphaFold2
code is Apache-2.0 (github.com/google-deepmind/alphafold); the AF2 model
parameters are CC-BY-4.0 with DeepMind's terms of use.
```

Third-party code, weights, and services the platform declares:

| Kind | Name | Provider | Licence | Terms | Info | Privacy |
| --- | --- | --- | --- | --- | --- | --- |
| weights | AlphaFold2 | Google DeepMind | CC-BY-4.0 | https://github.com/google-deepmind/alphafold#model-parameters-license | not stated | not stated |
| service | ColabFold MSA server (api.colabfold.com) | Steinegger Lab | not stated | not stated | https://github.com/sokrypton/ColabFold/wiki | not stated |

Notes the platform wrote inside the frontmatter, verbatim. These carry its own licence provenance and verification dates:

```
# README #model-parameters-license: "AlphaFold parameters … CC BY 4.0".
# SKILL.md body: "the public MMseqs2 server" (api.colabfold.com — user's
# sequence is POSTed there for MSA). verified 2026-06-30
# api.colabfold.com has no published ToS or privacy policy. The GitHub
# wiki is the closest data-use reference (info_url, not terms_url).
# Hostname in `name` so the user sees exactly where their sequence goes.
```

## boltz

| Field | Value |
| --- | --- |
| Directory | `0.1.41-release/skills/boltz/` |
| Declared name | `boltz` |
| Display name | not stated |
| Category | `biomodels` |
| Skill licence, as the platform states it | `Apache-2.0` |
| Declared requirements | `[gpu]` |
| Compute route declarations | frontmatter `requirements: [gpu]` |
| Version or pin | not stated. The frontmatter of `boltz/SKILL.md` carries no version, release, or commit field. |
| Files shipped | `SKILL.md` |
| `SKILL.md` SHA-256 | `4205a08363955f762f45a2065263de0dabedd898eba13460830b70b621b26700` |

Description, verbatim from `boltz/SKILL.md` frontmatter:

```
Structure prediction for protein, nucleic-acid, and small-molecule complexes
with Boltz-2 (Passaro & Wohlwend et al. 2025, github.com/jwohlwend/boltz).
Reach for this skill to validate designed binders against a target, to
co-fold a protein with a SMILES or CCD ligand, or to get an open-source
AlphaFold3 alternative with optional binding-affinity prediction.
```

Opening paragraph, verbatim from the `boltz/SKILL.md` body. This is where the skill names the tool and model it wraps:

```
Boltz-2 is the open-weights diffusion co-folder closest in surface to
AlphaFold3: a YAML describing protein, DNA, RNA, and ligand chains in, mmCIF
plus pTM/ipTM/pLDDT confidences out, with an optional small-molecule affinity
head. Among our four co-fold skills it is the default for binder-validation
campaigns — fully open MIT weights and the fastest sampler; pick `chai1` when
you want a second independent model for consensus, `openfold3` when AF3-faithful
settings matter, and `esmfold2` when you can live without an MSA. Code and
weights are MIT (PyPI `boltz`, github.com/jwohlwend/boltz).
```

Third-party code, weights, and services the platform declares:

| Kind | Name | Provider | Licence | Terms | Info | Privacy |
| --- | --- | --- | --- | --- | --- | --- |
| weights | Boltz-2 | not stated | MIT | https://github.com/jwohlwend/boltz/blob/main/LICENSE | not stated | not stated |
| service | ColabFold MSA server (api.colabfold.com) | Steinegger Lab | not stated | not stated | https://github.com/sokrypton/ColabFold/wiki | not stated |

Notes the platform wrote inside the frontmatter, verbatim. These carry its own licence provenance and verification dates:

```
# github.com/jwohlwend/boltz/blob/main/LICENSE: MIT (© 2024 Wohlwend, Corso,
# Passaro). verified 2026-06-30
# SKILL.md — `--use_msa_server` queries api.colabfold.com; the doc says to
# add it when a chain has no MSA. User's sequence is POSTed there. No
# published ToS — wiki is the closest data-use reference.
```

## borzoi

| Field | Value |
| --- | --- |
| Directory | `0.1.41-release/skills/borzoi/` |
| Declared name | `borzoi` |
| Display name | not stated |
| Category | `biomodels` |
| Skill licence, as the platform states it | `Apache-2.0` |
| Declared requirements | `[gpu]` |
| Compute route declarations | frontmatter `requirements: [gpu]` |
| Version or pin | not stated. The frontmatter of `borzoi/SKILL.md` carries no version, release, or commit field. |
| Files shipped | `SKILL.md` |
| `SKILL.md` SHA-256 | `a6b87d906c21eb51c141f7510b2e77ef8eab7127abb17a66467043408d92d702` |

Description, verbatim from `borzoi/SKILL.md` frontmatter:

```
Predict genome-wide functional tracks (RNA-seq, CAGE, DNase, ChIP) from DNA
sequence with Borzoi. Use this skill when:
(1) Scoring the regulatory effect of a variant on expression/accessibility,
(2) Generating predicted coverage tracks for a locus,
(3) Prioritizing non-coding variants by predicted track delta.
```

The body of `borzoi/SKILL.md` opens on a heading, table, or code block rather than a prose paragraph, so no opening paragraph is transcribed.

Third-party code, weights, and services the platform declares:

| Kind | Name | Provider | Licence | Terms | Info | Privacy |
| --- | --- | --- | --- | --- | --- | --- |
| weights | Borzoi (PyTorch port) | Calico Life Sciences | CC-BY-4.0 | not stated | https://huggingface.co/johahi/borzoi-replicate-0 | not stated |

Notes the platform wrote inside the frontmatter, verbatim. These carry its own licence provenance and verification dates:

```
# SKILL.md loads `johahi/borzoi-replicate-0` — a PyTorch port of Calico's
# Borzoi ("ported weights (with permission)"). The HuggingFace model card
# for that exact artifact states `License: cc-by-4.0`. Calico's CODE repo is
# Apache-2.0, but the weights the skill downloads carry CC-BY-4.0. The model
# card is where the license is declared (info_url — not a ToU page).
# verified 2026-06-30
```

## chai1

| Field | Value |
| --- | --- |
| Directory | `0.1.41-release/skills/chai1/` |
| Declared name | `chai1` |
| Display name | `Chai-1` |
| Category | `biomodels` |
| Skill licence, as the platform states it | `Apache-2.0` |
| Declared requirements | `[gpu]` |
| Compute route declarations | frontmatter `requirements: [gpu]` |
| Version or pin | not stated. The frontmatter of `chai1/SKILL.md` carries no version, release, or commit field. |
| Files shipped | `SKILL.md` |
| `SKILL.md` SHA-256 | `14c7b111475cfb2df0c1ba4de8427b09099439417101bc443fe52e8c80eff539` |

Description, verbatim from `chai1/SKILL.md` frontmatter:

```
Structure prediction for protein, nucleic-acid, and small-molecule complexes
with the Chai-1 foundation model (Chai Discovery 2024,
github.com/chaidiscovery/chai-lab). Reach for this skill to predict an
antibody-antigen or protein-ligand complex from a single FASTA, to re-fold
designed binders as an AlphaFold-multimer alternative, or to drive
co-folding from Python for batched campaigns on a GPU.
```

Opening paragraph, verbatim from the `chai1/SKILL.md` body. This is where the skill names the tool and model it wraps:

```
Chai-1 is an all-atom diffusion co-folder in the same family as Boltz-2 and
AlphaFold3: a multi-entity FASTA in, mmCIF plus pTM/ipTM/pLDDT out, with
protein, RNA, DNA, and SMILES-ligand chains all first-class. It and `boltz`
cover the same surface; running both and keeping designs that pass either is a
common consensus filter, and Chai's Python entry point makes it the easier of
the two to embed in a loop. Code and weights are Apache-2.0 — commercial use
including drug discovery is explicitly permitted
(github.com/chaidiscovery/chai-lab).
```

Third-party code, weights, and services the platform declares:

| Kind | Name | Provider | Licence | Terms | Info | Privacy |
| --- | --- | --- | --- | --- | --- | --- |
| weights | Chai-1 | Chai Discovery | Apache-2.0 | https://github.com/chaidiscovery/chai-lab/blob/main/LICENSE | not stated | not stated |
| service | ColabFold MSA server (api.colabfold.com) | Steinegger Lab | not stated | not stated | https://github.com/sokrypton/ColabFold/wiki | not stated |

Notes the platform wrote inside the frontmatter, verbatim. These carry its own licence provenance and verification dates:

```
# github.com/chaidiscovery/chai-lab/blob/main/LICENSE: Apache-2.0
# (relicensed at v0.4.0). verified 2026-06-30
# SKILL.md — `--use-msa-server` / `use_msa_server=True` sends the sequence
# to the public ColabFold MMseqs2 server. No published ToS — wiki is the
# closest data-use reference.
```

## compute-env-setup

| Field | Value |
| --- | --- |
| Directory | `0.1.41-release/skills/compute-env-setup/` |
| Declared name | `compute-env-setup` |
| Display name | not stated |
| Category | not stated |
| Skill licence, as the platform states it | `Apache-2.0` |
| Declared requirements | not stated |
| Compute route declarations | not stated in `compute-env-setup/SKILL.md` frontmatter, and the directory ships no `provider.json` |
| Version or pin | not stated. The frontmatter of `compute-env-setup/SKILL.md` carries no version, release, or commit field. |
| Files shipped | `SKILL.md`, `references/envs_reference.md` |
| `SKILL.md` SHA-256 | `33a9053330f66e76f2cbbc782271804247aae7dfff224e8ebeba7d58ba1e1d7a` |

Description, verbatim from `compute-env-setup/SKILL.md` frontmatter:

```
Set up a compute environment on a remote provider so Claude Science jobs can run there. Covers direct SSH/conda hosts, Slurm clusters, container-via-bridge runners, and managed-API providers (Modal, GCP, RunPod). Use when standing up a new provider, porting an env to a different backend, adding a tool that needs its own software stack, or wiring weight caches. Triggers on "new compute provider", "set up env on", "port env to", "build GPU image", "weight cache", "compute_details", "conda env on the box", "apptainer on slurm".
```

The body of `compute-env-setup/SKILL.md` opens on a heading, table, or code block rather than a prose paragraph, so no opening paragraph is transcribed.

The platform declares no `metadata.third_party` block for this skill, so third-party code, weights, and services are not stated. Checked `compute-env-setup/SKILL.md` frontmatter.

## diffdock

| Field | Value |
| --- | --- |
| Directory | `0.1.41-release/skills/diffdock/` |
| Declared name | `diffdock` |
| Display name | `DiffDock` |
| Category | `biomodels` |
| Skill licence, as the platform states it | `Apache-2.0` |
| Declared requirements | `[gpu]` |
| Compute route declarations | frontmatter `requirements: [gpu]` |
| Version or pin | not stated. The frontmatter of `diffdock/SKILL.md` carries no version, release, or commit field. |
| Files shipped | `SKILL.md`, `references/workflows.md` |
| `SKILL.md` SHA-256 | `6c3acc1710209b8daf2322e6103043c3487412153e2455deab6abd851ad8efb6` |

Description, verbatim from `diffdock/SKILL.md` frontmatter:

```
Predict small-molecule binding poses with DiffDock-L (Corso et al. 2023/2024,
github.com/gcorso/DiffDock) — blind diffusion docking that places a ligand
into a protein pocket without a predefined search box and ranks the samples
with a learned confidence model. Reach for this skill to dock a
SMILES or SDF against a PDB, to generate ranked 3D poses for a small
fragment library, or to get a starting pose for downstream rescoring.
DiffDock predicts geometry, not affinity.
```

Opening paragraph, verbatim from the `diffdock/SKILL.md` body. This is where the skill names the tool and model it wraps:

```
DiffDock-L is a blind pose predictor: given a protein structure and a ligand,
it samples ligand placements over the whole surface with a diffusion model and
ranks them with a separately trained confidence head. The confidence score
correlates with pose correctness, not with binding free energy — DiffDock does
not predict whether or how tightly the ligand binds, so for hit triage you
still pair it with a scorer (GNINA, MM-GBSA) or with `boltz`'s affinity head.
For protein–protein and nucleic-acid co-folding, route to `boltz` or `chai1`.
Code and weights are MIT (github.com/gcorso/DiffDock).
```

Third-party code, weights, and services the platform declares:

| Kind | Name | Provider | Licence | Terms | Info | Privacy |
| --- | --- | --- | --- | --- | --- | --- |
| weights | DiffDock-L | not stated | MIT | https://github.com/gcorso/DiffDock/blob/main/LICENSE | not stated | not stated |

Notes the platform wrote inside the frontmatter, verbatim. These carry its own licence provenance and verification dates:

```
# github.com/gcorso/DiffDock/blob/main/LICENSE: MIT (© 2022 Corso, Stärk,
# Jing). verified 2026-06-30
```

## esmfold2

| Field | Value |
| --- | --- |
| Directory | `0.1.41-release/skills/esmfold2/` |
| Declared name | `esmfold2` |
| Display name | `ESMFold2` |
| Category | `biomodels` |
| Skill licence, as the platform states it | `Apache-2.0` |
| Declared requirements | `[gpu]` |
| Compute route declarations | frontmatter `requirements: [gpu]` |
| Version or pin | not stated. The frontmatter of `esmfold2/SKILL.md` carries no version, release, or commit field. |
| Files shipped | `SKILL.md`, `references/design-hook.md`, `references/esmc.md` |
| `SKILL.md` SHA-256 | `28e625bab51a5d33e278b5ee6f304d76afa35c424fd919e1914d649df7600175` |

Description, verbatim from `esmfold2/SKILL.md` frontmatter:

```
Biohub ESMFold2 / ESMFold2-Fast all-atom co-folding (Candido et al. 2026,
github.com/Biohub/esm). Single-sequence and MSA modes; protein, DNA, RNA,
ligand (CCD/SMILES), modified residues. FoldBench Ab-Ag 50-55%, PPI 70-77%
DockQ-pass. Also covers the ESMC-{300M,600M,6B} protein language models from
the same release: masked-LM logits, hidden states, mutation scoring, contact
prediction, and the SAE interpretability head. MIT-licensed weights on
HuggingFace org `biohub`. Use this skill when: (1) Predicting complex
structures with single-sequence input, (2) Validating designed binders with
ESMFold2-Fast, (3) Running ESMFold2 with MSA input, (4) Getting ESMC
embeddings or per-residue mutation scores, (5) Choosing kernel backend and
sampling-step settings for paper-faithful throughput.
```

Opening paragraph, verbatim from the `esmfold2/SKILL.md` body. This is where the skill names the tool and model it wraps:

```
All-atom diffusion co-folding from the Biohub ESM release (2026). ESMFold2 =
48 pair layers with MSA support; ESMFold2-Fast = 24 layers, single-sequence
only, ~1.7x faster.
```

Third-party code, weights, and services the platform declares:

| Kind | Name | Provider | Licence | Terms | Info | Privacy |
| --- | --- | --- | --- | --- | --- | --- |
| weights | ESMFold2 / ESMC | Biohub | MIT | https://github.com/Biohub/esm/blob/main/LICENSE.md | not stated | not stated |

Notes the platform wrote inside the frontmatter, verbatim. These carry its own licence provenance and verification dates:

```
# SKILL.md body: "**License:** MIT (code github.com/Biohub/esm + weights HF
# `biohub/*`)"
# github.com/Biohub/esm/blob/main/LICENSE.md: MIT (© 2026 Chan Zuckerberg
# Biohub, Inc.). verified 2026-06-30
```

## evo2

| Field | Value |
| --- | --- |
| Directory | `0.1.41-release/skills/evo2/` |
| Declared name | `evo2` |
| Display name | `Evo 2` |
| Category | `biomodels` |
| Skill licence, as the platform states it | `Apache-2.0` |
| Declared requirements | `[gpu]` |
| Compute route declarations | frontmatter `requirements: [gpu]` |
| Version or pin | not stated. The frontmatter of `evo2/SKILL.md` carries no version, release, or commit field. |
| Files shipped | `SKILL.md` |
| `SKILL.md` SHA-256 | `48909aa815d5c2c1b42c81796b93667223689e3d13c7de5568e59c76ae328fd7` |

Description, verbatim from `evo2/SKILL.md` frontmatter:

```
Score, embed, and generate DNA sequences with Evo 2, a long-context genomic
foundation model. Use this skill when:
(1) Computing per-nucleotide or per-sequence likelihoods for variant effect
    scoring,
(2) Embedding genomic windows for downstream classification,
(3) Generating DNA conditioned on a prefix,
(4) Scoring regulatory or coding regions across species.
```

The body of `evo2/SKILL.md` opens on a heading, table, or code block rather than a prose paragraph, so no opening paragraph is transcribed.

Third-party code, weights, and services the platform declares:

| Kind | Name | Provider | Licence | Terms | Info | Privacy |
| --- | --- | --- | --- | --- | --- | --- |
| weights | Evo 2 | Arc Institute | Apache-2.0 | https://github.com/ArcInstitute/evo2/blob/main/LICENSE | not stated | not stated |

Notes the platform wrote inside the frontmatter, verbatim. These carry its own licence provenance and verification dates:

```
# github.com/ArcInstitute/evo2/blob/main/LICENSE: Apache-2.0 boilerplate.
# HuggingFace model cards `arcinstitute/evo2_{40b_base,20b}` declare
# `license: apache-2.0`. verified 2026-06-30
```

## fair-esm2

| Field | Value |
| --- | --- |
| Directory | `0.1.41-release/skills/fair-esm2/` |
| Declared name | `fair-esm2` |
| Display name | `ESM-2` |
| Category | `biomodels` |
| Skill licence, as the platform states it | `Apache-2.0` |
| Declared requirements | `[gpu]` |
| Compute route declarations | frontmatter `requirements: [gpu]` |
| Version or pin | not stated. The frontmatter of `fair-esm2/SKILL.md` carries no version, release, or commit field. |
| Files shipped | `SKILL.md` |
| `SKILL.md` SHA-256 | `da5729e9014848de10296149541ccac30696144ba14c28f003de4cfdd9771480` |

Description, verbatim from `fair-esm2/SKILL.md` frontmatter:

```
Embed proteins with Meta AI's ESM-2 (`fair-esm` package). Use this skill
when: (1) Extracting per-residue or per-sequence embeddings for downstream
ML, (2) Masked-LM likelihood / mutation effect scoring, (3) Contact
prediction from a sequence.
```

Opening paragraph, verbatim from the `fair-esm2/SKILL.md` body. This is where the skill names the tool and model it wraps:

```
ESM-2 code and weights are MIT (Meta AI, github.com/facebookresearch/esm).
```

Third-party code, weights, and services the platform declares:

| Kind | Name | Provider | Licence | Terms | Info | Privacy |
| --- | --- | --- | --- | --- | --- | --- |
| weights | ESM-2 | Meta AI | MIT | https://github.com/facebookresearch/esm/blob/main/LICENSE | not stated | not stated |

Notes the platform wrote inside the frontmatter, verbatim. These carry its own licence provenance and verification dates:

```
# github.com/facebookresearch/esm/blob/main/LICENSE: MIT (© Meta Platforms,
# Inc. and affiliates). verified 2026-06-30
```

## figure-composer

| Field | Value |
| --- | --- |
| Directory | `0.1.41-release/skills/figure-composer/` |
| Declared name | `figure-composer` |
| Display name | not stated |
| Category | not stated |
| Skill licence, as the platform states it | `Apache-2.0` |
| Declared requirements | not stated |
| Compute route declarations | not stated in `figure-composer/SKILL.md` frontmatter, and the directory ships no `provider.json` |
| Version or pin | not stated. The frontmatter of `figure-composer/SKILL.md` carries no version, release, or commit field. |
| Files shipped | `SKILL.md`, `kernel.py` |
| `SKILL.md` SHA-256 | `37f4dc3fb964a527e51d305e2d545a63d43e84009011780b9a411f8696fdebd1` |

Description, verbatim from `figure-composer/SKILL.md` frontmatter:

```
Compose one publication-grade multi-panel figure. Entry from a one-line claim + data refs, OR from an existing figure via `derive_outline(png)`. Runs a per-figure loop: outline (12-col grid, per-panel ask + label_budget) → fan-out one sub-agent per panel (each loads `figure-style`) → tile + stamp letters → adversarial composite review with two-tier feedback (Tier-1 outline_revisions / Tier-2 per-panel violations) → regen affected panels, ≤3 rounds. Loads panel_task / compose_figure / compose_crops / composite_review_task / derive_outline into the kernel. For one standalone plot use `figure-style`; for whole-paper figure ordering use `paper-narrative`.
```

Opening paragraph, verbatim from the `figure-composer/SKILL.md` body. This is where the skill names the tool and model it wraps:

```
**Step 0.** Load `figure-style` alongside this skill — that is the
design rules (and `apply_figure_style()` + helpers). Panel sub-agents
will load it independently; you need it in context to write the outline and
review the composite. Sub-agents run as the default profile and acquire the
rules by loading the skill.
```

The platform declares no `metadata.third_party` block for this skill, so third-party code, weights, and services are not stated. Checked `figure-composer/SKILL.md` frontmatter.

## figure-style

| Field | Value |
| --- | --- |
| Directory | `0.1.41-release/skills/figure-style/` |
| Declared name | `figure-style` |
| Display name | not stated |
| Category | not stated |
| Skill licence, as the platform states it | `Apache-2.0` |
| Declared requirements | not stated |
| Compute route declarations | not stated in `figure-style/SKILL.md` frontmatter, and the directory ships no `provider.json` |
| Version or pin | not stated. The frontmatter of `figure-style/SKILL.md` carries no version, release, or commit field. |
| Files shipped | `SKILL.md`, `kernel.py` |
| `SKILL.md` SHA-256 | `cdf7f4e13bb69044a256a77379a9e0465ba2901b50a9359196efa8e5243ebc04` |

Description, verbatim from `figure-style/SKILL.md` frontmatter:

```
Publication-grade figure correctness and legibility rules for final-deliverable figures — not every plot. Quick look or iterating on the analysis (EDA scatters, sanity-check histograms)? Plot plainly without this skill. Producing a figure that ships — report, paper, export, or kept artifact — load this skill first and call `apply_figure_style()` — sets a role-mapped font-size ladder, outward ticks, frameless legends, and 300-dpi output. The skill is a checklist, not a house look: data fidelity (claim-titles tested against every row, excluded data never enters summaries), label economy (floor and ceiling), color threading, chart-choice-by-data-shape, layout, and a render-then-verify QA loop (bbox collision + per-panel perceptual check). Ships helpers: focal_palette, bar_with_points, strip_with_median, end_of_line_labels, panel_letter, set_frame, panel_crops. For multi-panel figures load `figure-composer`; for whole-paper figure arc load `paper-narrative`.
```

Opening paragraph, verbatim from the `figure-style/SKILL.md` body. This is where the skill names the tool and model it wraps:

```
*A checklist for correct, legible, internally-consistent scientific figures. This
skill does not impose a visual house style — frame, font, and palette are
parameters. Once loaded, call `apply_figure_style()` before plotting.*
```

The platform declares no `metadata.third_party` block for this skill, so third-party code, weights, and services are not stated. Checked `figure-style/SKILL.md` frontmatter.

## ligandmpnn

| Field | Value |
| --- | --- |
| Directory | `0.1.41-release/skills/ligandmpnn/` |
| Declared name | `ligandmpnn` |
| Display name | `LigandMPNN` |
| Category | `biomodels` |
| Skill licence, as the platform states it | `Apache-2.0` |
| Declared requirements | not stated |
| Compute route declarations | not stated in `ligandmpnn/SKILL.md` frontmatter, and the directory ships no `provider.json` |
| Version or pin | not stated. The frontmatter of `ligandmpnn/SKILL.md` carries no version, release, or commit field. |
| Files shipped | `SKILL.md` |
| `SKILL.md` SHA-256 | `004f25fe7caab394424fb7e2660b9428ad7079d0b9ff1d573079ab6340938396` |

Description, verbatim from `ligandmpnn/SKILL.md` frontmatter:

```
Inverse-fold a backbone with ligand, nucleic-acid, and metal context using
LigandMPNN (Dauparas et al. 2023, github.com/dauparas/LigandMPNN). Reach for
this skill to redesign the residues lining a binding pocket around a bound
small molecule or cofactor, to design metal-coordinating sites where the
geometry must be respected, or to get threaded designed-sequence PDBs out of
any MPNN run.
```

Opening paragraph, verbatim from the `ligandmpnn/SKILL.md` body. This is where the skill names the tool and model it wraps:

```
LigandMPNN extends the ProteinMPNN graph with non-protein atoms — small
molecules, nucleic acids, and metals are visible to the network — so it is the
right inverse-folding tool whenever the design surface includes a bound ligand
or cofactor that vanilla `proteinmpnn` would ignore. The same `run.py` is also
the most convenient runner for the other MPNN families because, unlike the
original ProteinMPNN script, it threads designs back onto the input structure
and writes PDBs alongside the FASTA. Code and weights are MIT
(github.com/dauparas/LigandMPNN). The model is small enough to run on CPU —
for a handful of designs on one structure that is seconds and usually faster
than dispatching, so the normal path is local with
`pip install torch numpy biopython ProDy ml_collections dm-tree`; a GPU helps
for batched campaigns.
```

Third-party code, weights, and services the platform declares:

| Kind | Name | Provider | Licence | Terms | Info | Privacy |
| --- | --- | --- | --- | --- | --- | --- |
| weights | LigandMPNN | not stated | MIT | https://github.com/dauparas/LigandMPNN/blob/main/LICENSE | not stated | not stated |

Notes the platform wrote inside the frontmatter, verbatim. These carry its own licence provenance and verification dates:

```
# github.com/dauparas/LigandMPNN/blob/main/LICENSE: MIT (© 2024 Justas
# Dauparas). verified 2026-06-30
```

## managed-model-endpoints

| Field | Value |
| --- | --- |
| Directory | `0.1.41-release/skills/managed-model-endpoints/` |
| Declared name | `managed-model-endpoints` |
| Display name | not stated |
| Category | not stated |
| Skill licence, as the platform states it | `Apache-2.0` |
| Declared requirements | not stated |
| Compute route declarations | not stated in `managed-model-endpoints/SKILL.md` frontmatter, and the directory ships no `provider.json` |
| Version or pin | not stated. The frontmatter of `managed-model-endpoints/SKILL.md` carries no version, release, or commit field. |
| Files shipped | `SKILL.md` |
| `SKILL.md` SHA-256 | `48cb2bf810b3c54d43ab8384effe019c725e1faf14161464eb8fabf7e8d8d47a` |

Description, verbatim from `managed-model-endpoints/SKILL.md` frontmatter:

```
Register a model service in the managed family — a local model server container the daemon starts/stops on demand, or a remote upstream model API (https). Read the runbook, allocate a port (local only), compose idempotent start/stop scripts (local only), register once. Load when the user wants a model service available for inference, or when list_compute shows managed endpoints.
```

Opening paragraph, verbatim from the `managed-model-endpoints/SKILL.md` body. This is where the skill names the tool and model it wraps:

```
A **managed model endpoint** is a model service the **daemon** owns: you
register it **once**, then every `compute_provider` cell against it just
works — the daemon swaps the resident model off the device (one model at a
time, via the resident's own approved stop), runs your approved start
script, waits for the readiness route, then runs your cell, streaming its
lifecycle progress into the cell as it goes. You never run the container
runtime yourself, never poll readiness in cells, and never see the
credential value. Two verbs: `register()` (asks the user once) and ordinary
inference cells.
```

The platform declares no `metadata.third_party` block for this skill, so third-party code, weights, and services are not stated. Checked `managed-model-endpoints/SKILL.md` frontmatter.

## openfold3

| Field | Value |
| --- | --- |
| Directory | `0.1.41-release/skills/openfold3/` |
| Declared name | `openfold3` |
| Display name | `OpenFold3` |
| Category | `biomodels` |
| Skill licence, as the platform states it | `Apache-2.0` |
| Declared requirements | `[gpu]` |
| Compute route declarations | frontmatter `requirements: [gpu]` |
| Version or pin | not stated. The frontmatter of `openfold3/SKILL.md` carries no version, release, or commit field. |
| Files shipped | `SKILL.md` |
| `SKILL.md` SHA-256 | `eef2b03a2c7e1c4b94cd29bb27f3e02499a77be5c414a494f3d78e9c7cc443de` |

Description, verbatim from `openfold3/SKILL.md` frontmatter:

```
Structure prediction using OpenFold3, an open-weights PyTorch reproduction of
AlphaFold3 from the AlQuraishi Lab.
Use this skill when predicting protein/nucleic-acid/ligand complex
structures with an Apache-2.0-licensed AF3 reimplementation.
```

The body of `openfold3/SKILL.md` opens on a heading, table, or code block rather than a prose paragraph, so no opening paragraph is transcribed.

Third-party code, weights, and services the platform declares:

| Kind | Name | Provider | Licence | Terms | Info | Privacy |
| --- | --- | --- | --- | --- | --- | --- |
| weights | OpenFold3 | OpenFold Consortium | Apache-2.0 | https://github.com/aqlaboratory/openfold-3/blob/main/LICENSE | not stated | not stated |
| service | ColabFold MSA server (api.colabfold.com) | Steinegger Lab | not stated | not stated | https://github.com/sokrypton/ColabFold/wiki | not stated |

Notes the platform wrote inside the frontmatter, verbatim. These carry its own licence provenance and verification dates:

```
# github.com/aqlaboratory/openfold-3/blob/main/LICENSE: Apache-2.0. HF model
# card and gated prompt confirm. verified 2026-06-30
# SKILL.md — `--use-msa-server` DEFAULTS to true (MSA server is
# api.colabfold.com), so the sequence leaves the machine unless opted out.
# No published ToS — wiki is the closest data-use reference.
```

## proteinmpnn

| Field | Value |
| --- | --- |
| Directory | `0.1.41-release/skills/proteinmpnn/` |
| Declared name | `proteinmpnn` |
| Display name | `ProteinMPNN` |
| Category | `biomodels` |
| Skill licence, as the platform states it | `Apache-2.0` |
| Declared requirements | not stated |
| Compute route declarations | not stated in `proteinmpnn/SKILL.md` frontmatter, and the directory ships no `provider.json` |
| Version or pin | not stated. The frontmatter of `proteinmpnn/SKILL.md` carries no version, release, or commit field. |
| Files shipped | `SKILL.md` |
| `SKILL.md` SHA-256 | `9873f4f800d457a6b6a7139df133e8b4b259694c9433e188ca551929373f47b8` |

Description, verbatim from `proteinmpnn/SKILL.md` frontmatter:

```
Inverse-fold a protein backbone (PDB structure) into amino-acid sequence with
ProteinMPNN (Dauparas et al. 2022, github.com/dauparas/ProteinMPNN). Reach
for this skill to run sequence design on RFdiffusion backbones, to redesign
one chain of a PDB while holding interface residues fixed, or to generate a
temperature-swept set of sequences for downstream folding.
```

Opening paragraph, verbatim from the `proteinmpnn/SKILL.md` body. This is where the skill names the tool and model it wraps:

```
ProteinMPNN is the default inverse-folding step in the binder pipeline: a
message-passing network that sees backbone geometry only, so it is the right
choice when the design surface is protein–protein and the wrong one as soon as
a ligand, nucleic acid, or metal is part of the interface — `ligandmpnn` adds
those atoms to the graph with a near-identical CLI, and `solublempnn` swaps in
weights trained on soluble structures for an expression-biased prior. Code and
weights are MIT (github.com/dauparas/ProteinMPNN). The model is small enough
to run on CPU — for a handful of sequences on one backbone that is seconds and
usually faster than dispatching a remote job; a GPU helps for batched
campaigns (hundreds of backbones or large `--num_seq_per_target`). Either way
the repo is cloned in-job — there is no PyPI dist and the checkpoints are
bundled in the repo.
```

Third-party code, weights, and services the platform declares:

| Kind | Name | Provider | Licence | Terms | Info | Privacy |
| --- | --- | --- | --- | --- | --- | --- |
| weights | ProteinMPNN | not stated | MIT | https://github.com/dauparas/ProteinMPNN/blob/main/LICENSE | not stated | not stated |

Notes the platform wrote inside the frontmatter, verbatim. These carry its own licence provenance and verification dates:

```
# github.com/dauparas/ProteinMPNN/blob/main/LICENSE: MIT (© 2022 Justas
# Dauparas). verified 2026-06-30
```

## remote-compute-modal

| Field | Value |
| --- | --- |
| Directory | `0.1.41-release/skills/remote-compute-modal/` |
| Declared name | `remote-compute-modal` |
| Display name | not stated |
| Category | not stated |
| Skill licence, as the platform states it | `Apache-2.0` |
| Declared requirements | not stated |
| Compute route declarations | ships `provider.json` |
| Version or pin | not stated. The frontmatter of `remote-compute-modal/SKILL.md` carries no version, release, or commit field. |
| Files shipped | `SKILL.md`, `env-setup.md`, `envs/chemistry_gpu.py`, `envs/esmfold2_gpu.py`, `envs/genomics_evo2_gpu.py`, `envs/md_openmm_gpu.py`, `envs/proteomics_boltz_gpu.py`, `envs/proteomics_gpu.py`, `envs/proteomics_jax_gpu.py`, `envs/proteomics_openfold_gpu.py`, `envs/proteomics_rfd_diffdock_gpu.py`, `envs/singlecell_gpu.py`, `provider.json`, `provider.py`, `requirements.lock` |
| `SKILL.md` SHA-256 | `6a29e9961d99efec68c940edbcdf20ceb1fc04c91c3a7747af16791531575e92` |

Description, verbatim from `remote-compute-modal/SKILL.md` frontmatter:

```
Run GPU jobs on the user's own Modal account via host.compute.create('modal', provider_params={...}) — the create→submit→wait_for_notification flow, the compute_provider kernel for env setup, image/volume resolution, and the two approval cards. Load once you've decided to dispatch to Modal.
```

Opening paragraph, verbatim from the `remote-compute-modal/SKILL.md` body. This is where the skill names the tool and model it wraps:

```
print(host.skills.read('remote-compute-modal', 'envs/proteomics_jax_gpu.py')['content'])
```

The platform declares no `metadata.third_party` block for this skill, so third-party code, weights, and services are not stated. Checked `remote-compute-modal/SKILL.md` frontmatter.

## remote-compute-ssh

| Field | Value |
| --- | --- |
| Directory | `0.1.41-release/skills/remote-compute-ssh/` |
| Declared name | `remote-compute-ssh` |
| Display name | not stated |
| Category | not stated |
| Skill licence, as the platform states it | `Apache-2.0` |
| Declared requirements | not stated |
| Compute route declarations | not stated in `remote-compute-ssh/SKILL.md` frontmatter, and the directory ships no `provider.json` |
| Version or pin | not stated. The frontmatter of `remote-compute-ssh/SKILL.md` carries no version, release, or commit field. |
| Files shipped | `SKILL.md` |
| `SKILL.md` SHA-256 | `6b0a1bbb5543bb001e7dac65b5d47159cca278364a6625aef26deda555a624f3` |

Description, verbatim from `remote-compute-ssh/SKILL.md` frontmatter:

```
Submit→wait_for_notification→collect-outputs workflow for the user's SSH/SLURM hosts. Load once you've decided to dispatch remote.
```

Opening paragraph, verbatim from the `remote-compute-ssh/SKILL.md` body. This is where the skill names the tool and model it wraps:

```
c = host.compute.create('<cluster>')            # bare name; 'ssh:<cluster>' works too
job = c.submit_job(
    intent='<tool> on <input> — 1 GPU, ~10 min',
    command='''#SBATCH --gres=gpu:1
```

The platform declares no `metadata.third_party` block for this skill, so third-party code, weights, and services are not stated. Checked `remote-compute-ssh/SKILL.md` frontmatter.

## scgpt

| Field | Value |
| --- | --- |
| Directory | `0.1.41-release/skills/scgpt/` |
| Declared name | `scgpt` |
| Display name | `scGPT` |
| Category | `biomodels` |
| Skill licence, as the platform states it | `Apache-2.0` |
| Declared requirements | `[gpu]` |
| Compute route declarations | frontmatter `requirements: [gpu]` |
| Version or pin | not stated. The frontmatter of `scgpt/SKILL.md` carries no version, release, or commit field. |
| Files shipped | `SKILL.md` |
| `SKILL.md` SHA-256 | `cfc7161b3dccb7fe2ac7d46642b649a209c4eaddf9512f4faae27d36eb29bc97` |

Description, verbatim from `scgpt/SKILL.md` frontmatter:

```
Embed and annotate single-cell expression data with scGPT, a foundation model
for single-cell biology. Use this skill when:
(1) Producing cell embeddings from an AnnData for clustering/integration,
(2) Zero-shot or fine-tuned cell-type annotation,
(3) Gene-level representation for perturbation/GRN tasks.

For probabilistic single-cell models (scVI etc.), use the scvi-tools
library.
```

The body of `scgpt/SKILL.md` opens on a heading, table, or code block rather than a prose paragraph, so no opening paragraph is transcribed.

Third-party code, weights, and services the platform declares:

| Kind | Name | Provider | Licence | Terms | Info | Privacy |
| --- | --- | --- | --- | --- | --- | --- |
| weights | scGPT | Wang Lab (University of Toronto) | not stated | not stated | https://github.com/bowang-lab/scGPT | not stated |

Notes the platform wrote inside the frontmatter, verbatim. These carry its own licence provenance and verification dates:

```
# scGPT checkpoints are distributed as unlabeled Google Drive directories
# (linked from github.com/bowang-lab/scGPT); the repo LICENSE (MIT) covers
# the CODE, and no source states a weights license. Per the sourcing rule:
# leave `license` absent. Repo root is a README, not a terms page —
# info_url. verified 2026-06-30
```

## scvi-tools

| Field | Value |
| --- | --- |
| Directory | `0.1.41-release/skills/scvi-tools/` |
| Declared name | `scvi-tools` |
| Display name | `scvi-tools` |
| Category | not stated |
| Skill licence, as the platform states it | `Apache-2.0` |
| Declared requirements | `[gpu]` |
| Compute route declarations | frontmatter `requirements: [gpu]` |
| Version or pin | not stated. The frontmatter of `scvi-tools/SKILL.md` carries no version, release, or commit field. |
| Files shipped | `SKILL.md`, `kernel.py` |
| `SKILL.md` SHA-256 | `34866479168c8ef7f8e48c2a8e62b35263079af3b749ac35d96ed9748bcfa917` |

Description, verbatim from `scvi-tools/SKILL.md` frontmatter:

```
Probabilistic single-cell RNA-seq with scvi-tools — scVI for a
batch-corrected latent space, scANVI for semi-supervised label transfer,
and Bayesian differential expression. Reach for this skill to integrate
scRNA-seq batches, embed cells for clustering, transfer annotations from a
reference onto a query, or score differentially expressed genes per cluster.
For spatial deconvolution / mapping use the cell2location, DestVI, or
Tangram methods instead.
```

Opening paragraph, verbatim from the `scvi-tools/SKILL.md` body. This is where the skill names the tool and model it wraps:

```
scvi-tools (Gayoso et al. 2022, github.com/scverse/scvi-tools, BSD-3-Clause)
wraps a family
of deep generative models for single-cell omics. The scRNA-seq core is **scVI**
(unsupervised batch-corrected latent embedding) and **scANVI** (scVI + a
classifier head for semi-supervised cell-type label transfer). Both expect
**raw integer UMI counts** and emit a low-dimensional `X_scVI` / `X_scANVI`
that drops into the scanpy neighbors → leiden → umap pipeline.
```

The platform declares an empty `metadata.third_party` block for this skill, so third-party code, weights, and services are not stated. Checked `scvi-tools/SKILL.md` frontmatter.

## self-awareness

| Field | Value |
| --- | --- |
| Directory | `0.1.41-release/skills/self-awareness/` |
| Declared name | `self-awareness` |
| Display name | not stated |
| Category | not stated |
| Skill licence, as the platform states it | `Apache-2.0` |
| Declared requirements | not stated |
| Compute route declarations | not stated in `self-awareness/SKILL.md` frontmatter, and the directory ships no `provider.json` |
| Version or pin | not stated. The frontmatter of `self-awareness/SKILL.md` carries no version, release, or commit field. |
| Files shipped | `SKILL.md` |
| `SKILL.md` SHA-256 | `52a2ada64113301301bb16441eb7375f96dc186fffbd0dec0f85c4938291ddb3` |

Description, verbatim from `self-awareness/SKILL.md` frontmatter:

```
Claude Science's own session database schema and SDK surface for introspection via host.query(). Load this when you need to query your own conversation history, token usage, cost accounting, execution log, or artifact metadata beyond what host.frames()/host.artifacts() provide — e.g. "how many tokens has this session used", "what was my last tool call", "list every file I've written", "where are messages stored", "what tables can I query", "inspect frames.context_data", or any time you're about to PRAGMA-probe the Claude Science metadata DB to discover its schema.
```

Opening paragraph, verbatim from the `self-awareness/SKILL.md` body. This is where the skill names the tool and model it wraps:

```
`host.query(sql, params=[], limit=None, df=False)` runs read-only SQLite
against Claude Science's own metadata DB. It is only available via the **`repl`
tool** (not `python`/`r`). Results are automatically scoped to the current
project, so `SELECT * FROM frames` returns only frames in this project. The
`repl` tool is stdlib-only — `df=True` returns the raw dict there (use
`json.dump(..., open("handoff/q.json","w"))` and load in a `python` cell if
you want pandas).
```

The platform declares no `metadata.third_party` block for this skill, so third-party code, weights, and services are not stated. Checked `self-awareness/SKILL.md` frontmatter.

## solublempnn

| Field | Value |
| --- | --- |
| Directory | `0.1.41-release/skills/solublempnn/` |
| Declared name | `solublempnn` |
| Display name | `SolubleMPNN` |
| Category | `biomodels` |
| Skill licence, as the platform states it | `Apache-2.0` |
| Declared requirements | not stated |
| Compute route declarations | not stated in `solublempnn/SKILL.md` frontmatter, and the directory ships no `provider.json` |
| Version or pin | not stated. The frontmatter of `solublempnn/SKILL.md` carries no version, release, or commit field. |
| Files shipped | `SKILL.md` |
| `SKILL.md` SHA-256 | `588a8cd3e0781e76a0a86bb87c5f3f37b087ceba760bcaf0d28f9e9835351ec8` |

Description, verbatim from `solublempnn/SKILL.md` frontmatter:

```
Inverse-fold a backbone with SolubleMPNN — ProteinMPNN retrained on a
soluble-PDB subset (Dauparas et al. 2022) — for sequences biased toward
cytosolic expression and reduced aggregation. Reach for this skill when designs from vanilla
ProteinMPNN are aggregating or going to inclusion bodies, when redesigning a
membrane-adjacent fold for soluble expression, or when an E. coli expression
screen is the next step.
```

Opening paragraph, verbatim from the `solublempnn/SKILL.md` body. This is where the skill names the tool and model it wraps:

```
SolubleMPNN is not a separate package — it is the ProteinMPNN architecture
retrained on a soluble-PDB subset, which shifts the output distribution away
from the surface hydrophobics that the full-PDB model happily places (because
many of them are buried at crystallographic or membrane interfaces in the
training set). Reach for it when the goal is soluble yield in a heterologous
host; stick with `proteinmpnn` when native-like recovery matters more, since
the soluble prior trades a few points of recovery for the surface bias. Code
and weights are MIT (github.com/dauparas/ProteinMPNN, `soluble_model_weights`;
also exposed via github.com/dauparas/LigandMPNN). The model is small enough to
run on CPU — for a handful of sequences on one backbone that is seconds and
usually faster than dispatching; a GPU helps for batched campaigns. Either way
the repo is cloned in-job (no PyPI dist; checkpoints bundled).
```

Third-party code, weights, and services the platform declares:

| Kind | Name | Provider | Licence | Terms | Info | Privacy |
| --- | --- | --- | --- | --- | --- | --- |
| weights | SolubleMPNN | not stated | MIT | https://github.com/dauparas/ProteinMPNN/blob/main/LICENSE | not stated | not stated |

Notes the platform wrote inside the frontmatter, verbatim. These carry its own licence provenance and verification dates:

```
# github.com/dauparas/ProteinMPNN/blob/main/LICENSE: MIT (© 2022 Justas
# Dauparas). soluble_model_weights ship in the same repo with no separate
# license. verified 2026-06-30
```

## using-model-endpoint

| Field | Value |
| --- | --- |
| Directory | `0.1.41-release/skills/using-model-endpoint/` |
| Declared name | `using-model-endpoint` |
| Display name | not stated |
| Category | not stated |
| Skill licence, as the platform states it | `Apache-2.0` |
| Declared requirements | not stated |
| Compute route declarations | ships `provider.json` |
| Version or pin | not stated. The frontmatter of `using-model-endpoint/SKILL.md` carries no version, release, or commit field. |
| Files shipped | `SKILL.md`, `provider.json`, `provider.py`, `requirements.lock` |
| `SKILL.md` SHA-256 | `2ee613c4530f85e21db322b4e15d510a1800c228dee4c010a2849e55f852af00` |

Description, verbatim from `using-model-endpoint/SKILL.md` frontmatter:

```
Call a registered model endpoint over its native HTTP API from the endpoint's scoped inference kernel (BASE_URL preloaded). Load once a task needs predictions from a registered model endpoint.
```

The body of `using-model-endpoint/SKILL.md` opens on a heading, table, or code block rather than a prose paragraph, so no opening paragraph is transcribed.

The platform declares no `metadata.third_party` block for this skill, so third-party code, weights, and services are not stated. Checked `using-model-endpoint/SKILL.md` frontmatter.

## File inventory

Every file the inventory hash covers, with its own SHA-256.

| Path | SHA-256 | Bytes |
| --- | --- | --- |
| `THIRD_PARTY_LICENSES.md` | `7777d62188aa6b84fd316687eb26d69b454a1a9d8fb1929067895cb7f68db745` | 69,842 |
| `alphafold2/SKILL.md` | `562520a36043c09f6e740b28a097e1e11a68b73bcc3c4c7cb0e269a644b4427a` | 4,965 |
| `boltz/SKILL.md` | `4205a08363955f762f45a2065263de0dabedd898eba13460830b70b621b26700` | 5,305 |
| `borzoi/SKILL.md` | `a6b87d906c21eb51c141f7510b2e77ef8eab7127abb17a66467043408d92d702` | 3,910 |
| `chai1/SKILL.md` | `14c7b111475cfb2df0c1ba4de8427b09099439417101bc443fe52e8c80eff539` | 4,615 |
| `compute-env-setup/SKILL.md` | `33a9053330f66e76f2cbbc782271804247aae7dfff224e8ebeba7d58ba1e1d7a` | 17,880 |
| `compute-env-setup/references/envs_reference.md` | `c53c6d4a4ecb630ea54579389fd8aefaac8c31d806b311eddb70df88277759f9` | 18,249 |
| `diffdock/SKILL.md` | `6c3acc1710209b8daf2322e6103043c3487412153e2455deab6abd851ad8efb6` | 4,476 |
| `diffdock/references/workflows.md` | `4f4d780abb02650e475c6e2392db44dc615ecc9c0676a6442847118481c980f9` | 1,984 |
| `esmfold2/SKILL.md` | `28e625bab51a5d33e278b5ee6f304d76afa35c424fd919e1914d649df7600175` | 10,208 |
| `esmfold2/references/design-hook.md` | `f100def8706fcadd5ffba599137b910ed7b2896b23e46473f81c414f5335f75a` | 1,452 |
| `esmfold2/references/esmc.md` | `62ce8706e7493a537af64e1550c6203d36e2c29a36a46fc2d3afc1b87681abce` | 3,052 |
| `evo2/SKILL.md` | `48909aa815d5c2c1b42c81796b93667223689e3d13c7de5568e59c76ae328fd7` | 5,411 |
| `fair-esm2/SKILL.md` | `da5729e9014848de10296149541ccac30696144ba14c28f003de4cfdd9771480` | 4,961 |
| `figure-composer/SKILL.md` | `37f4dc3fb964a527e51d305e2d545a63d43e84009011780b9a411f8696fdebd1` | 6,773 |
| `figure-composer/kernel.py` | `0d6f04d14e637028aa58a5be276edac85e6118890abf65b6821a3b048c56d87a` | 13,778 |
| `figure-style/SKILL.md` | `cdf7f4e13bb69044a256a77379a9e0465ba2901b50a9359196efa8e5243ebc04` | 17,233 |
| `figure-style/kernel.py` | `8d16286025719a395e744c5bedcefd1592b1b8e3450263dd1ea7807ffec70122` | 12,457 |
| `ligandmpnn/SKILL.md` | `004f25fe7caab394424fb7e2660b9428ad7079d0b9ff1d573079ab6340938396` | 6,217 |
| `managed-model-endpoints/SKILL.md` | `48cb2bf810b3c54d43ab8384effe019c725e1faf14161464eb8fabf7e8d8d47a` | 10,718 |
| `openfold3/SKILL.md` | `eef2b03a2c7e1c4b94cd29bb27f3e02499a77be5c414a494f3d78e9c7cc443de` | 7,378 |
| `proteinmpnn/SKILL.md` | `9873f4f800d457a6b6a7139df133e8b4b259694c9433e188ca551929373f47b8` | 4,587 |
| `remote-compute-modal/SKILL.md` | `6a29e9961d99efec68c940edbcdf20ceb1fc04c91c3a7747af16791531575e92` | 49,756 |
| `remote-compute-modal/env-setup.md` | `0d323eb9bb47c21af9df2a5f20a876b50b748080f11e43a73032aba4f7cf6a59` | 15,530 |
| `remote-compute-modal/envs/chemistry_gpu.py` | `0b929572f2b4f6823486d8635156145da45ca310b0fd0e4b1f09397fe08ad548` | 950 |
| `remote-compute-modal/envs/esmfold2_gpu.py` | `b84e2babc270ab7fe8f4edde09bc4f60c67257397c4cd12389a48768edac5e31` | 3,644 |
| `remote-compute-modal/envs/genomics_evo2_gpu.py` | `c3f3e4703998f061176abc2715b74060f939b84000b024fcfe6e35ec5f1c5055` | 2,095 |
| `remote-compute-modal/envs/md_openmm_gpu.py` | `7dd15673229a614b5f58f4e3f6cbb324ad33bc3020568ee46e876082feed3126` | 1,436 |
| `remote-compute-modal/envs/proteomics_boltz_gpu.py` | `fc3f36390514cc778f230bc05de910564ee026c3b710997a0ea4d7d59c1008b9` | 1,311 |
| `remote-compute-modal/envs/proteomics_gpu.py` | `765076a3076fcccc9dce7669d8caf6a861f51cda40708065cafd5540e877edbe` | 2,450 |
| `remote-compute-modal/envs/proteomics_jax_gpu.py` | `f862b09b9319fa17ffde4b588c209a2a1429a786d6599da023927fc60e4d7f0f` | 1,643 |
| `remote-compute-modal/envs/proteomics_openfold_gpu.py` | `20562138429cbaa54853164adaf689abbcc37998e9299c3bfae71107c66039ff` | 3,049 |
| `remote-compute-modal/envs/proteomics_rfd_diffdock_gpu.py` | `ff53e00ef83847d4793efca6e1986a58c64a876f1d54e26f2ac886c5fa3bbef5` | 4,103 |
| `remote-compute-modal/envs/singlecell_gpu.py` | `ba5c49a11097a0248edde20b5f06d7b9d390e4cdd5f2bf29655d3454c6a542ac` | 3,156 |
| `remote-compute-modal/provider.json` | `3ec5afe85350f4cb82b4370cfa3bd9c1d08695d34335689ab5d6278dd74e54b0` | 342 |
| `remote-compute-modal/provider.py` | `2f453df1917690b888f12816f416f260dd8e44e59084a7e22ab3e31eea362470` | 36,978 |
| `remote-compute-modal/requirements.lock` | `6974b03b3293732cc6d1b9183848264a4ff403048e23df79d9061e57905e27c1` | 72,358 |
| `remote-compute-ssh/SKILL.md` | `6b0a1bbb5543bb001e7dac65b5d47159cca278364a6625aef26deda555a624f3` | 27,510 |
| `scgpt/SKILL.md` | `cfc7161b3dccb7fe2ac7d46642b649a209c4eaddf9512f4faae27d36eb29bc97` | 5,454 |
| `scvi-tools/SKILL.md` | `34866479168c8ef7f8e48c2a8e62b35263079af3b749ac35d96ed9748bcfa917` | 11,270 |
| `scvi-tools/kernel.py` | `4abd8a39ac92ef75222bc5b57f80d7bd0f00b6899d58863c2311ef0eadfcb6c4` | 1,982 |
| `self-awareness/SKILL.md` | `52a2ada64113301301bb16441eb7375f96dc186fffbd0dec0f85c4938291ddb3` | 15,079 |
| `solublempnn/SKILL.md` | `588a8cd3e0781e76a0a86bb87c5f3f37b087ceba760bcaf0d28f9e9835351ec8` | 4,277 |
| `using-model-endpoint/SKILL.md` | `2ee613c4530f85e21db322b4e15d510a1800c228dee4c010a2849e55f852af00` | 1,754 |
| `using-model-endpoint/provider.json` | `306e4111dbcb642635e020d1caf94c75f24f18526401e85d43e6658c37a517d9` | 187 |
| `using-model-endpoint/provider.py` | `4508e5affd203575edd564398bf4c7bb12ddea75128488f2cdbb380ded8f2f1a` | 3,628 |
| `using-model-endpoint/requirements.lock` | `8ec4de650015bb499b861f2950b8af52156f622f610406760e261c4868ff3fa5` | 1,580 |

