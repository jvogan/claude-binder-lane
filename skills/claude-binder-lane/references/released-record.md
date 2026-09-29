# The released record

Read this when comparing a run against the published campaign.

**Scope.** The median table below carries fourteen of the release's sixteen design codes. Two are out of scope for this skill and are deliberately absent. [Published targets](published-targets.md#scope) states the rule.

## Contents

- [Reading the released record](#reading-the-released-record)
- [Checking a run against the campaign](#checking-a-run-against-the-campaign)

## Reading the released record

The published release carries the campaign designs and a per-design summary table.

Filter the table before reading. The table has 65 columns. Columns 1 to 12 carry identity, provenance, and design method. Columns 43 to 65 carry the structure directory, per-predictor scores, and epitope residues. These 35 computational columns are the only columns this skill uses. Columns 13 to 42 originate outside the computational pipeline. They do not appear in this skill, in examples, in test fixtures, or in generated run metadata.

Four properties of the released table determine what a run can check against it:

The summary table cannot reproduce the campaign ranking. It carries the rank, generator, sequence-design method, optimization round count, and per-predictor `ipsae_min` and `sc_dockq`. It omits `rank_zscore`, `final_score`, `pose_dockq`, `pose_PASS`, instrument mask, seed count, root backbone ID, and cluster ID. The score behind the rank column cannot be recomputed from this table.

Several provenance columns contain values for only a subset of designs. These include local composition perplexity score, sequence-model log-likelihood, parent design ID, epitope or hotspot specifications, and final-gate instruments. A null entry indicates unknown status. The record does not indicate whether per-design restraints executed on every design.

Round counts range from 0 to 26 and record design ancestry depth. Two designs from one root backbone can carry different counts. A low count does not indicate that its target ran few rounds.

Scope differs between the prompt and the release. The multi-target protocol names 14 targets. The released table covers 1,440 designs across 16 targets and three campaigns. Single-target campaigns ran on targets omitted from the multi-target prompt. Every figure taken from the release retains its source denominator.

## Checking a run against the campaign

Published per-design scores provide a numerical baseline for replication. Execute the pipeline on a published campaign target, score candidates across the three published ranking modes, and compare the resulting distribution against published rows for that target.

Compare raw values. The `rank_zscore` metric is transductive and applies only within its source pool. Two pools can produce identical z-scores from different raw values. Compare `ipsae_min` and `sc_dockq`, which do not depend on the pool.

Compare matching targets, stoichiometries, and predictors. Published ipSAE values are not comparable across targets, stoichiometries, or predictors. Each row in the median table provides a benchmark for one target on one predictor arm at a stated stoichiometry.

Median `ipsae_min` per target, computed from the published table:

| Target | Designs | `ef2fast` | `ef2full` | `ptxv2` |
|---|---|---|---|---|
| PD-L1 | 90 | 0.879 | 0.866 | 0.915 |
| TrkA | 90 | 0.848 | 0.843 | 0.900 |
| IL-7Ra | 90 | 0.828 | 0.846 | 0.880 |
| TREM2 | 90 | 0.815 | 0.804 | 0.849 |
| RBX1 | 90 | 0.754 | 0.764 | 0.828 |
| BBF-14 | 90 | 0.729 | 0.693 | 0.769 |
| VEGF-A | 90 | 0.716 | 0.726 | 0.745 |
| 15-PGDH | 30 | 0.691 | 0.665 | 0.784 |
| MBP | 90 | 0.671 | 0.713 | 0.764 |
| Mature GDF-8 | 120 | 0.594 | 0.613 | 0.573 |
| EGFR | 90 | 0.542 | 0.738 | 0.675 |
| TNFa | 150 | 0.364 | 0.541 | 0.675 |
| Latent GDF-8 | 60 | 0.293 | 0.329 | 0.662 |
| Cas9 | 90 | 0.000 | 0.000 | 0.018 |

The published campaign did not score every target well. A pipeline that returns high scores across all targets indicates an error.

The published reference evaluated all three modes. Predictors agree near the top of the table and diverge on lower-scoring targets. On EGFR, median `ipsae_min` is 0.542 on `ef2fast` and 0.738 on `ef2full`. On TNFa, the medians are 0.364 on `ef2fast`, 0.541 on `ef2full`, and 0.675 on `ptxv2`. On latent GDF-8, the medians are 0.293 on `ef2fast`, 0.329 on `ef2full`, and 0.662 on `ptxv2`. A single-arm evaluation on a low-scoring target does not establish replication.

PD-L1 provides the primary reference target. It contains 90 designs, a median `ipsae_min` of 0.879 on `ef2fast`, and 0.915 on `ptxv2`. Rows exist in both the multi-target and single-target campaigns. PD-L1 is not the release ceiling. One out-of-scope target carries a higher median on all three modes and a higher minimum score, 0.541 against 0.065, so do not read a PD-L1 median as the best the release achieved. The validation gate requires a non-antibody literature control at native stoichiometry.

Cas9 serves as the negative control. Its median `ipsae_min` is 0.000 on both ESMFold2 arms and 0.018 on Protenix v2. The value is 0.000 for 68 designs on `ef2fast`, 70 designs on `ef2full`, and 40 designs on `ptxv2` out of 90 total designs. The deployed profile runs the two ESMFold2 modes and records Protenix v2 as unrun.

Two properties apply to the Cas9 benchmark. Cas9 co-folds evaluate the ribonucleoprotein complex with loaded guide RNA. Because ipSAE excludes RNA tokens, a near-zero score reflects interface contact failure on protein chains. The pose metric does not correlate with the interface confidence metric on Cas9: median `sc_dockq_ptxv2` is 0.472, while median `ipsae_min_ptxv2` is 0.018. Evaluate the control using ipSAE.

The top Cas9 design reaches 0.576 on `ef2fast`. Compare full distributions across the cohort.

Replication criteria must be set before starting a run. The requirements appear in [open-questions.md](open-questions.md).
