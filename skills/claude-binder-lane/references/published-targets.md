# Published targets

The released multi-target prompt names fourteen targets and states how to choose
their constructs and epitopes. This page transcribes twelve of those targets.
Read it when a campaign intends to follow that prompt's target setup. Anthropic's
[public study post](https://www.anthropic.com/research/Claude-accelerates-protein-design)
reports 15 targets attempted and binders found for 14; those are outcome counts
from the reported study, not the prompt's target-list length.

Source: the [released multi-target binder design prompt](https://huggingface.co/datasets/Anthropic/claude-protein-binder-design/blob/d442eeb/prompts/prompts/multi_target_binder_design_prompt.md) at commit `d442eeb`, read on 2026-09-16. The dataset is licensed CC BY 4.0, and the table and rules below are transcribed from it and attributed to Anthropic. Its target table and its construct and epitope rules are transcribed below. The [published campaign comparison](published-campaign-comparison.md) compares the shipped profile against the rest of that prompt.

## Scope

This page transcribes twelve of the prompt's fourteen targets. For a target
outside this table, read its construct and epitope requirements in the
[released prompt](https://huggingface.co/datasets/Anthropic/claude-protein-binder-design/blob/d442eeb/prompts/prompts/multi_target_binder_design_prompt.md).
The [published results](published-results.md) and [released record](released-record.md)
pages present fourteen of the release's sixteen design codes. Each page names
the denominator used for its figures.

## The targets in scope

`ECD` is the extracellular domain. The oligomeric state and notes are the release's own, and the `#` column keeps the release's numbering, so it skips the out-of-scope rows.

| # | Target | UniProt | Organism | Oligomeric state and notes |
| --- | --- | --- | --- | --- |
| 1 | EGFR / HER1 / ErbB1 ECD | P00533 | H. sapiens | Monomer |
| 2 | PD-L1 (CD274 / B7-H1) | Q9NZQ7 | H. sapiens | Monomer |
| 3 | IL-7Ra / CD127 ECD | P16871 | H. sapiens | Monomer |
| 4 | BBF-14 | none, de novo 8-stranded beta-barrel, ref PDB 9HAG, NCBI taxid 32630 | synthetic | Monomer |
| 6 | MBP (maltose-binding protein, malE) | P0AEX9 | E. coli K-12 | Monomer. Not human myelin basic protein |
| 7 | Cas9 (SpCas9) | Q99ZW2 | S. pyogenes M1 | Monomer. The functional unit is the 1:1 SpCas9:sgRNA RNP |
| 8 | TREM2 ECD | Q9NZC2 | H. sapiens | Monomer |
| 9 | TrkA / NTRK1 ECD | P04629 | H. sapiens | Monomer |
| 10 | TNFa | P01375 | H. sapiens | Homotrimer (C3) |
| 11 | VEGF-A | P15692 | H. sapiens | Homodimer (C2, disulfide-linked, antiparallel) |
| 12 | Myostatin / GDF-8 | O14793 | H. sapiens | Homodimer (C2, disulfide-linked). Counter-target GDF-11 (O95390, homodimer C2). Binders must be selective against GDF-11, and counter-screening and selectivity optimization are required parts of this target |
| 13 | RBX1 (ROC1) | P62877 | H. sapiens | Monomer. The RING domain coordinates three structural Zn2+, and the native context is a heteromeric Cullin-RING ligase |

The packaged campaign template names `Q9NZQ7`, which is target 2.

## How the release picks the construct

A review of the target's biology comes before design. Four things are confirmed and recorded in the design sheet's target metadata before any design is generated or scored for a target:

- the oligomeric state the release evaluated the target in
- the constitutive cofactors and ligands, meaning structural metals, nucleic-acid partners, and glycans
- the exact construct, meaning the residue range, tags, and fusion context
- which deposited structures represent that state with the epitope ordered

The scoring construct matches that system rather than a convenience crop. When the per-target dossier flags a cofactor as fold-required or interface-required, at least one ranking arm has to represent it.

Where several deposited structures exist, designing against more than one is allowed rather than overfitting a single deposition. Prioritize the structure that best matches that construct and state.

## How the release picks the epitope

Prioritize biologically relevant interfaces and epitopes already explored in the design literature. A novel epitope is allowed when you hold a differentiated hypothesis. Prefer functional epitopes where a miniprotein binder can plausibly achieve a measurable mechanism of action. The epitope choice accounts for the non-protein context identified in the review, meaning metals, ligands, and membranes.

Cross-species reactivity is a secondary objective, pursued only without compromising affinity or hit rate against the primary antigen. A second epitope per target is encouraged once the primary is on track and budget allows, and it is not required.

Binders are single-chain miniproteins of 50 to 120 residues. The release permits 35 to 160 where epitope geometry motivates it, with the rationale recorded.

## Published controls

The release also publishes the target and control sequences it used, as `data/controls/` in the same dataset, with every record in `targets_and_controls.fasta`. A campaign reproducing a target does not have to choose its own positive control. Read the record rather than pick one.

These are the panels the release measured its own designs against. The validation gate asks for a known literature binder scoring clearly above negatives, and these name the binder the release actually used per target. Whether the same molecule also served as the in-silico gate control is not stated in the prompt, so treat the match as the release's own choice of literature binder rather than as a transcribed gate file.

| Target | Published positive control |
| --- | --- |
| BBF-14 | BBF-14 binder 4, published de novo binder, PDB 9HAC |
| Cas9 | AcrIIA26, as expressed |
| EGFR | Human EGF, P01133 971-1023; and a de novo anti-EGFR scFv from a public binder-design competition |
| IL-7Ra | Human IL-7, P13232 mature chain region; and a published de novo IL-7Ra binder |
| MBP | DARPin off7, PDB 1SVX; and an anti-MBP scFv derived from monoclonal antibody B48 |
| PD-L1 | A published de novo PD-L1 binder; and the human PD-1 extracellular domain, Q15116 |
| RBX1 | Human cullin-1 C-terminal domain 411-776, Q13616; and a de novo RBX1 binder from a public competition |
| TNFa | Adalimumab variable domains as an scFv; and the anti-TNF VHH domain of ozoralizumab, residues 1-115 |
| TREM2 | A de novo TREM2 binder from a public competition; and an anti-TREM2 scFv, PDB 6YMQ |
| TrkA | Human beta nerve growth factor mature chain, P01138 122-241; and a de novo TrkA binder from a public competition |
| VEGF-A | A published de novo VEGF-A binder; and human VEGFR-1 Ig-like domain 2, P17948 |

The controls directory covers a slightly different roster from the design table. It carries 15-PGDH and Latent GDF-8, and the target table above names Myostatin / GDF-8. Read the directory for the control roster and the table above for the design roster.

### The Cas9 record

The Cas9 rows are specific, and a campaign reproducing target 7 should read them before choosing a construct or a control.

- **Antigen.** SpCas9 residues 2 to 1368 at Q99ZW2, flowed as a ribonucleoprotein with a single-guide RNA. A second record carries the full-length 1 to 1368 reference sequence for the plate set.
- **Guide.** The 85-nucleotide sgRNA of PDB 4ZT0 chain B, spacer `GGCGCAUAAAGAUGAGACGC`, supplied as a custom synthesis of that sequence. The release requested that guide specifically.
- **Positive control.** AcrIIA26, as expressed.

**The release records a metric pathology on this target.** On the Cas9 RNP, ipSAE excludes the RNA tokens and comes out exactly zero for a large share of designs: 44 percent on Protenix v2, 58 percent on OpenDDE, and 76 to 83 percent for the other seven RNP predictors, across the 90 designs. Boltz-2 and OpenFold3 are both in that last group. ipSAE_min is the release's primary ranking term, so on this target it is silent for most candidates and a ranking built on it alone will not separate them. Plan the metric before the spend, and read `data/docs/DATA_NOTES.md` for the note itself.

The release also folded an apo comparison form, `cas9_apoprotein`, across all ten predictors, so an apo-versus-RNP question on this target has a published counterpart rather than needing invention.

AcrIIA4 is not the published control and 5VW1 is not the published guide. Both are defensible choices under the epitope rules above, and both are substitutions from the release's own record. Say which you are using.

## The construct each target was folded as

The release publishes one co-folding construct per target, in `data/docs/insilico_target_constructs.fasta` and `data/tables/insilico/target_constructs.parquet`, with the methodology in `data/docs/INSILICO.md`. Every co-fold and design model used that one construct. This answers the crop question directly: read the row rather than derive a construct.

| Target | Construct id | Chains folded |
| --- | --- | --- |
| 15-PGDH | `15pgdh_dimer_266` | 2 protein copies of 266, plus NAD |
| BBF-14 | `bbf14_110` | 1 protein of 110 |
| Cas9 | `cas9_rnp_1368_sgrna98` | 1 protein of 1368 and 1 nucleic chain of 98 |
| EGFR | `egfr_621` | 1 protein of 621 |
| Mature GDF-8 | `mature_gdf8_dimer_109` | 2 protein copies of 109 |
| Latent GDF-8 | `latent_gdf8_dimer_352` | 2 protein copies of 352 |
| IL-7Ra | `il7ra_193` | 1 protein of 193 |
| MBP | `mbp_370` | 1 protein of 370 |
| PD-L1 | `pdl1_115` | 1 protein of 115 |
| RBX1 | `rbx1_108` | 1 protein of 108, plus three zinc ions |
| TNFa | `tnfa_trimer_157` | 3 protein copies of 157 |
| TREM2 | `trem2_156` | 1 protein of 156 |
| TrkA | `trka_101` | 1 protein of 101 |
| VEGF-A | `vegfa_dimer_94` | 2 protein copies of 94 |

The stoichiometry token travels with the construct: `1to1` for a monomeric target, `1to2` on a dimer, `1to3` on the TNF-alpha trimer, `2to2` and `3to3` for one binder per protomer, and `rnp` for one binder with SpCas9 plus sgRNA.

Cas9 was folded at full length with its guide. There is no truncation and no domain crop in the released construct, so a Cas9 crop is a substitution whatever motivates it.

## What the release ran, and on what

The broader released in-silico evaluation ran ten predictors, all template-free,
with five seeds per design and stoichiometry: Protenix v2, ESMFold2 full,
ESMFold2 fast, AlphaFold-Multimer v3, Boltz-2, Chai-1, OpenFold3, OpenDDE v1,
RoseTTAFold3, and AlphaFold3 code run with OpenFold3 weights. An unpaired target
MSA was staged per construct for MSA-capable arms; ESMFold2-Fast runs
single-sequence. The binder runs as a single sequence. The prompt's default
**ranking instrument uses three arms**: ESMFold2-Fast, ESMFold2-Full, and
Protenix v2. The other predictors belong to the broader evaluation and are
not all part of the ranking instrument; the prompt names eligible substitutes
for an unavailable or unvalidated arm. Only RoseTTAFold3,
the AlphaFold3-code arm, and Boltz-2 are deterministic at fixed input and seed.

Cofactors were supplied to every predictor that accepts them. The release names one exclusion, and it is instructive: AlphaFold-Multimer v3 folds without cofactors and was not run on the Cas9 RNP. Every other predictor was, ESMFold2 in both modes included. An arm that cannot carry the cofactor is dropped from that target rather than run blind, and no arm here was treated as protein-only by assumption.

Epitopes are published too. `design_summary.epitope_residues` lists the target residues within 5 angstroms of a binder heavy atom for each design, in the numbering of the co-fold construct, with per-residue contact detail in `data/tables/insilico/epitope_residue_contacts.parquet`. A campaign reproducing a target can read the epitope the release actually hit rather than derive one.

[Published results](published-results.md) carries what each target actually produced: hit rates, measured binding, ipSAE-zero rates and the generator mix per campaign. The release's design roster is sixteen codes against the prompt table's fourteen, and that page transcribes the fourteen in scope.

## Reading this against your own campaign

Three campaign shapes are distinct, and decision 2 records which one a run is.

**Reproduce a published target.** Take the row above, build the construct its notes describe, and select the epitope by the rules above. A target's note is load-bearing. Cas9 is a monomer whose functional unit is the RNP, so a Cas9 construct that drops the sgRNA is not the construct this release names, and [target inputs](target-inputs.md) covers declaring a kept nucleic acid chain.

**Reproduce the method on a target of your own.** The pipeline shape, the filters, and the ranking form carry over. The published thresholds do not, because they are frozen per target at a validation gate. Say so in decision 2 and make no fidelity claim on the target.

**Substitute both.** Nothing here applies except as a template. Record the reasons.

An orthologue is a substitution, not a reproduction. Target 7 is SpCas9 from S. pyogenes M1 at accession Q99ZW2. A different Cas9 orthologue answers a neighbouring question and its thresholds do not transfer.
