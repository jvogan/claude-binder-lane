# Published results

The release publishes its outcomes, and this page transcribes the part a campaign reads before it runs. Read it before decision 1 when you are choosing which target to reproduce, and before any rescoring arm when you are choosing which published designs to compare against. [Published targets](published-targets.md) carries the prompt's target table, construct rules, and controls. This page carries the design table those rules produced.

Source: `data/tables/design_summary.csv` in the [released dataset](https://huggingface.co/datasets/Anthropic/claude-protein-binder-design), read from `main` at revision `9e1b816` on 2026-09-16. The dataset is licensed CC BY 4.0 and every figure below is derived from it and attributed to Anthropic. Column definitions come from `data/docs/COLUMNS.md` at the same revision. [Published targets](published-targets.md) cites the prompt at commit `d442eeb`, which is a different revision of the same repository.

Every number here is derived, not quoted. [Re-derive the tables](#re-derive-the-tables) gives the command.

**Scope.** The tables below carry fourteen of the release's sixteen design codes. Two are out of scope for this skill and are deliberately absent. [Published targets](published-targets.md#scope) states the rule. Do not add the missing codes back from the upstream table.

## The design roster is 16, and the prompt table is 14

`design_summary.csv` holds **1,440 rows across 16 target codes**. The prompt names fourteen. Two differences account for the gap:

- **GDF-8 splits.** The prompt's target 12 is Myostatin / GDF-8. The design table carries `Mature GDF-8` and `Latent GDF-8` as separate codes, which `COLUMNS.md` describes as the mature growth factor and its latent pro-form.
- **15-PGDH is additional.** It appears in the design table, in `data/controls/`, and in the single-target prompts, and it is absent from the prompt's fourteen-target table.

The 16 codes also index `data/designs/<target>/`, so a folder name is a target code rather than a prompt row.

Each row is one ordered design. All 1,440 `uuid` values are distinct and all 1,440 sequences are distinct. These are the designs the release ordered and tested, not the pool it scored. The prompt asks for at least 50 scored backbones per starred method per target, so the scored pool behind this table is larger by orders of magnitude and the release does not publish it.

## The pool is the unit of comparison

Three columns define a pool, and mixing pools blends separate experiments:

- `campaign` takes three values: `multi_target`, `single_target`, and `single_target_supplementary`. The supplementary campaign covers TNFa alone.
- `design_model` takes two values: `Mythos Preview` and `Opus 4.8`. Both models appear in both main campaign types, so the campaign column alone does not identify the agent that produced a design.
- `target` takes the 16 codes above, of which fourteen are in scope here.

Those three columns partition the release's table into **48 pools of exactly 30 designs each**, and 48 times 30 is 1,440. Forty-two of those pools are in scope here. `rank` runs 1 to 30 within a pool, where 1 is the top-ranked submission.

A target carries between one and five pools. 15-PGDH has one. TNFa has five.

## Outcome by pool

`Hits / 30` counts `binder_final` true, which `COLUMNS.md` defines as the release's own final binder call on the human target, pooled across both vendors under a fixed rubric and checked against raw sensorgrams. `Best Kd` is the lowest `kd_nM_final` in the pool.

| Target | Campaign | Design model | Hits / 30 | Best Kd (nM) | Top generators |
| --- | --- | --- | --- | --- | --- |
| 15-PGDH | single | Mythos Preview | 1 | 65.16 | RFdiffusion3 15, BoltzGen 6, PXDesign 4 |
| BBF-14 | multi | Mythos Preview | 1 | 373.66 | PXDesign 8, Genie3 6, RFdiffusion3 4 |
| BBF-14 | multi | Opus 4.8 | 1 | 984.17 | RFdiffusion3 13, Genie3 12, RFdiffusion 5 |
| BBF-14 | single | Mythos Preview | 1 | 719.76 | PXDesign 6, Genie3 6, FreeBindCraft 5 |
| Cas9 | multi | Mythos Preview | 2 | 5.38 | PXDesign 14, BoltzGen 9, Proteina-Complexa 5 |
| Cas9 | multi | Opus 4.8 | 4 | 3.57 | FreeBindCraft 11, Proteina-Complexa 10, Genie3 8 |
| Cas9 | single | Mythos Preview | 4 | 24.34 | BoltzGen 15, RFdiffusion3 7, PXDesign 6 |
| EGFR | multi | Mythos Preview | 2 | 326.69 | PXDesign 15, FreeBindCraft 11, Proteina-Complexa 4 |
| EGFR | multi | Opus 4.8 | 1 | 539.61 | PXDesign 15, Genie3 9, RFdiffusion3 4 |
| EGFR | single | Mythos Preview | 7 | 17.31 | PXDesign 13, FreeBindCraft 10, RFdiffusion3 6 |
| IL-7Ra | multi | Mythos Preview | 18 | 13.90 | PXDesign 12, RFdiffusion3 8, FreeBindCraft 7 |
| IL-7Ra | multi | Opus 4.8 | 11 | 41.21 | PXDesign 14, RFdiffusion3 11, RFdiffusion 4 |
| IL-7Ra | single | Mythos Preview | 20 | 2.73 | BoltzGen 15, Proteina-Complexa 5, PXDesign 4 |
| Latent GDF-8 | single | Mythos Preview | 14 | 0.23 | Genie3 9, RFdiffusion3 8, RFdiffusion 7 |
| Latent GDF-8 | single | Opus 4.8 | 0 | - | RFdiffusion 12, RFdiffusion3 11, BoltzGen 3 |
| MBP | multi | Mythos Preview | 0 | - | PXDesign 13, RFdiffusion3 4, Genie3 4 |
| MBP | multi | Opus 4.8 | 0 | - | PXDesign 15, Genie3 12, RFdiffusion 3 |
| MBP | single | Mythos Preview | 0 | - | RFdiffusion3 15, PXDesign 11, RFdiffusion 3 |
| Mature GDF-8 | multi | Mythos Preview | withheld | - | BoltzGen 11, Genie3 5, FreeBindCraft 5 |
| Mature GDF-8 | multi | Opus 4.8 | withheld | - | Genie3 15, PXDesign 8, RFdiffusion3 7 |
| Mature GDF-8 | single | Mythos Preview | withheld | - | BoltzGen 9, Genie3 8, PXDesign 7 |
| Mature GDF-8 | single | Opus 4.8 | withheld | - | RFdiffusion3 15, PXDesign 9, BoltzGen 4 |
| PD-L1 | multi | Mythos Preview | 10 | 25.37 | PXDesign 15, RFdiffusion 5, FreeBindCraft 5 |
| PD-L1 | multi | Opus 4.8 | 2 | 270.45 | RFdiffusion3 13, Genie3 6, FoldCraft 5 |
| PD-L1 | single | Mythos Preview | 27 | 0.64 | FreeBindCraft 12, BoltzGen 6, PXDesign 5 |
| RBX1 | multi | Mythos Preview | 11 | 17.79 | PXDesign 15, Proteina-Complexa 6, RFdiffusion 5 |
| RBX1 | multi | Opus 4.8 | 5 | 29.35 | PXDesign 15, Genie3 9, RFdiffusion 6 |
| RBX1 | single | Mythos Preview | 12 | 7.00 | Genie3 15, PXDesign 9, RFdiffusion 4 |
| TNFa | multi | Mythos Preview | 0 | - | RFdiffusion3 10, BoltzGen 6, RFdiffusion 5 |
| TNFa | multi | Opus 4.8 | 8 | 2.10 | Genie3 15, RFdiffusion3 9, RFdiffusion 6 |
| TNFa | single | Mythos Preview | 0 | - | RFdiffusion3 15, PXDesign 9, Genie3 4 |
| TNFa | single | Opus 4.8 | 2 | 0.70 | PXDesign 15, RFdiffusion 9, RFdiffusion3 6 |
| TNFa | single supp | Opus 4.8 | 2 | 29.32 | PXDesign 30 |
| TREM2 | multi | Mythos Preview | 24 | 0.03 | FreeBindCraft 15, RFdiffusion3 7, BoltzGen 7 |
| TREM2 | multi | Opus 4.8 | 23 | 0.08 | PXDesign 15, Genie3 12, RFdiffusion3 3 |
| TREM2 | single | Mythos Preview | 25 | 0.07 | RFdiffusion3 13, PXDesign 9, FreeBindCraft 6 |
| TrkA | multi | Mythos Preview | 10 | 14.34 | RFdiffusion3 15, PXDesign 7, Proteina-Complexa 7 |
| TrkA | multi | Opus 4.8 | 2 | 271.52 | RFdiffusion3 14, PXDesign 10, FoldCraft 4 |
| TrkA | single | Mythos Preview | 8 | 10.87 | FreeBindCraft 14, RFdiffusion3 14, BoltzGen 2 |
| VEGF-A | multi | Mythos Preview | 13 | 63.07 | FreeBindCraft 15, RFdiffusion 7, Genie3 3 |
| VEGF-A | multi | Opus 4.8 | 20 | 3.17 | Genie3 15, BoltzGen 14, FoldCraft 1 |
| VEGF-A | single | Mythos Preview | 21 | 1.73 | RFdiffusion3 12, Proteina-Complexa 8, RFdiffusion 6 |

Four readings follow from that table.

**Hit rate spans the full range within one release.** PD-L1 single-target Mythos Preview returned 27 binders of 30 with a 0.64 nM best. PD-L1 multi-target Opus 4.8 returned 2 of 30 with a 270.45 nM best. Same target, same release, same fourteen-target prompt family.

**Generator mix is a pool property, not a release property.** FreeBindCraft supplies 12 of the 30 designs in PD-L1 single-target and 5 of 30 in PD-L1 multi-target Mythos Preview. Counting a generator across a whole target averages over pools that were built differently.

**MBP returned zero binders across all 90 tested designs.** Its three pools were tested and produced no hit. Treat MBP as a measured negative outcome.

**Mature GDF-8 carries no outcome.** `binder_final` and `kd_nM_final` are blank for all 120 of its designs, which `COLUMNS.md` attributes to outcome data the release does not include. A zero hit count on Mature GDF-8 is missing data. The `withheld` cells above mark it.

Column fill counts, as a check on what any comparison can use: `binder_final` 1,320 of 1,440, `kd_nM_final` 354, `adaptyv_kd_nM` 332, `twist_kd_nM` 325, `vendor_agreement` 1,320, `epitope_residues` 1,438.

## What the scoring columns mean

The table carries one `ipsae_min_<predictor>` and one `sc_dockq_<predictor>` column for each of ten predictors. Read both definitions before comparing your own numbers against them.

**`ipsae_min_<predictor>` is the maximum over the five seeds, not the mean.** `COLUMNS.md` states it as the maximum `ipsae_min` over the five seeds of that predictor, at the design campaign stoichiometry on the design target. A per-seed mean computed from your own rescore is a different statistic and will read low against this column.

**`sc_dockq_<predictor>` belongs to one seed.** It is the `sc_dockq` of the seed behind that maximum, so it is not a seed-averaged pose score.

`ipsae_min_*` is populated for all 1,440 rows. `sc_dockq_ef2fast` is populated for 1,274.

Predictor suffixes map to the ten predictors the release ran:

| Suffix | Predictor | Seed labels in the companion paths |
| --- | --- | --- |
| `ef2fast` | ESMFold2-Fast | 0 to 4 |
| `ef2full` | ESMFold2-Full | 0 to 4 |
| `ptxv2` | Protenix v2 | 0 to 4 |
| `odde` | OpenDDE v1 | 101 to 105 |
| `afm3` | AlphaFold-Multimer v3 | model index 1 to 5, or 0 to 4 on part of the design-target groups |
| `boltz2` | Boltz-2 | 0 to 4 |
| `chai1` | Chai-1 | 0 to 4 |
| `of3` | OpenFold3 | 0 to 4 |
| `rf3` | RoseTTAFold3 | 0 to 4 |
| `af3of3` | AlphaFold3 code with OpenFold3 weights | 1 to 5 |

The per-seed records sit in `data/tables/insilico/cofold_predictions.parquet`, which holds 113,550 rows.

## Where ipSAE returns zero

Each cell counts designs whose `ipsae_min` for that predictor is exactly zero. The prompt ranks on `ipSAE_min`, so a high count means the primary ranking term separates nothing on that target and arm.

| Target | N | ef2fast | ef2full | ptxv2 | odde | afm3 | boltz2 | chai1 | of3 | rf3 | af3of3 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 15-PGDH | 30 | 1 | 2 | 0 | 0 | 16 | 9 | 9 | 13 | 5 | 9 |
| BBF-14 | 90 | 1 | 0 | 1 | 0 | 23 | 3 | 8 | 27 | 26 | 15 |
| Cas9 | 90 | 68 | 70 | 40 | 52 | not run | 68 | 70 | 75 | 68 | 69 |
| EGFR | 90 | 9 | 0 | 0 | 1 | 19 | 10 | 27 | 17 | 6 | 11 |
| IL-7Ra | 90 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| Latent GDF-8 | 60 | 13 | 11 | 0 | 7 | 46 | 18 | 43 | 37 | 40 | 31 |
| MBP | 90 | 12 | 7 | 1 | 1 | 52 | 12 | 43 | 31 | 11 | 18 |
| Mature GDF-8 | 120 | 1 | 1 | 0 | 0 | 20 | 4 | 25 | 13 | 25 | 19 |
| PD-L1 | 90 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 2 | 2 |
| RBX1 | 90 | 1 | 2 | 0 | 0 | 12 | 0 | 12 | 1 | 1 | 1 |
| TNFa | 150 | 16 | 11 | 0 | 3 | 24 | 7 | 63 | 43 | 89 | 34 |
| TREM2 | 90 | 1 | 0 | 0 | 0 | 0 | 0 | 4 | 0 | 1 | 0 |
| TrkA | 90 | 1 | 0 | 0 | 0 | 2 | 0 | 6 | 3 | 3 | 2 |
| VEGF-A | 90 | 0 | 0 | 0 | 0 | 4 | 0 | 6 | 8 | 29 | 9 |

Three readings:

**Cas9 confirms the published pathology across every arm.** [Published targets](published-targets.md) records 44 percent on Protenix v2 and 58 percent on OpenDDE, which the `ptxv2` and `odde` cells reproduce at 40 and 52 of 90. The seven remaining arms sit between 68 and 75 of 90. The `afm3` cell reads `not run` because the release dropped AlphaFold-Multimer v3 on the RNP, and the table has no value there rather than a zero.

**The choice of arm changes the answer on most targets.** Protenix v2 returns a zero on 42 designs of 1,440, and 40 of those are Cas9. RoseTTAFold3 returns 314, of which 89 are TNFa. ESMFold2-Fast returns 128 and AlphaFold-Multimer v3 returns 280. An arm substitution moves the metric's coverage, so record the arm beside any ipSAE number.

**Three in-scope targets are clean on ESMFold2-Fast.** IL-7Ra, PD-L1, and VEGF-A record zero of them. PD-L1 records zero on seven of the ten arms.

## Method totals across the release

| Generator | Designs | Sequence design method | Designs |
| --- | --- | --- | --- |
| PXDesign | 387 | SolubleMPNN | 1,243 |
| RFdiffusion3 | 298 | Caliby (SolubleCaliby) | 114 |
| Genie3 | 213 | Native co-design by the structure model | 62 |
| BoltzGen | 158 | ProteinMPNN | 21 |
| FreeBindCraft (BindCraft) | 142 | | |
| RFdiffusion | 120 | | |
| Proteina-Complexa | 104 | | |
| FoldCraft | 14 | | |
| BoltzDesign1 | 2 | | |
| Protein Hunter | 2 | | |

Three names in the generator column sit outside the prompt's seven starred methods: FoldCraft, BoltzDesign1, and Protein Hunter. The prompt lists all three as permitted unstarred methods.

Binder length runs 50 to 120 residues with a mean of 81.9, inside the prompt's stated 50 to 120 band. `n_optimization_rounds` is 0 for 538 designs and reaches 26.

## Files this page reads

| Path | Size | Holds |
| --- | --- | --- |
| `data/tables/design_summary.csv` | 1,213,035 bytes | 1,440 designs, 65 columns, the source of every table here |
| `data/tables/design_summary.parquet` | 517,810 bytes | The same table |
| `data/designs/designs.fasta` | 272,603 bytes | The 1,440 binder sequences |
| `data/designs/<target>/<full_name>/` | Per design | Design model structure and co-folds |
| `data/tables/insilico/cofold_predictions.parquet` | 10,137,792 bytes | 113,550 per-seed predictor records |
| `data/tables/insilico/epitope_residue_contacts.parquet` | 386,426 bytes | Per-residue contacts behind `epitope_residues` |
| `data/tables/insilico/target_constructs.parquet` | 17,405 bytes | The construct each target was folded as |
| `data/docs/COLUMNS.md` | 106,603 bytes | Column definitions for every table |
| `data/docs/DATA_NOTES.md` | 19,821 bytes | The release's own caveats, including the Cas9 metric note |
| `data/docs/WETLAB.md` | 18,170 bytes | The binder-call rubric behind `binder_final` |

## Re-derive the tables

Fetch the table and count the pools yourself rather than trusting this page:

```bash
curl -sL -o design_summary.csv \
  https://huggingface.co/datasets/Anthropic/claude-protein-binder-design/resolve/main/data/tables/design_summary.csv

python3 - <<'PY'
import csv, collections
rows = list(csv.DictReader(open("design_summary.csv")))
pools = collections.Counter((r["target"], r["campaign"], r["design_model"]) for r in rows)
print(len(rows), "designs in", len(pools), "pools, sizes", set(pools.values()))
for key in sorted(pools):
    pool = [r for r in rows if (r["target"], r["campaign"], r["design_model"]) == key]
    hits = sum(1 for r in pool if r["binder_final"].strip().lower() == "true")
    tested = sum(1 for r in pool if r["binder_final"].strip() != "")
    print(key, hits, "of", tested)
PY
```

The command prints every code in the release, including the two this page leaves out of scope. Compare only against a target this page lists.

A revision later than `9e1b816` may change these counts. Re-run the command rather than editing a number in place.

## Reading this against your own campaign

Pick the pool before you pick the arm, and record both in decision 2.

**Reproducing a published target.** Name the pool, meaning the target, the campaign, and the design model together. A comparison against "the PD-L1 designs" spans three pools whose hit rates are 27, 10, and 2 of 30.

**Choosing a generator arm.** Read the top-generators column of the pool you chose. A generator with 3 designs in that pool gives you 3 counterparts, whatever its release-wide total is.

**Choosing a ranking arm.** Read the zero table for your target before committing to `ipSAE_min`. On Cas9 no available arm separates most of the pool, and the release's own apo comparison form is the published alternative.

**Comparing your rescore to a published column.** Take the maximum over your seeds, not the mean, because that is what the column holds. Record which predictor build you ran, since the release names only RoseTTAFold3, the AlphaFold3-code arm, and Boltz-2 as deterministic at fixed input and seed.

**Claiming a hit rate.** 354 designs of 1,440 carry a `kd_nM_final`, and 120 carry no outcome at all. State the denominator you used.
