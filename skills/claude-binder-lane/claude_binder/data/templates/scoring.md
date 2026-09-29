# Binder Lane Measurement And Ranking Contract

The lane stores raw measurements before it calculates gates or ranks. Every final observation uses the key:

```text
target_id × candidate_id × predictor × uniform-rescore × seed
```

The final table contains one row for every registered key, including failed predictions. Duplicate, unexpected, or mixed-phase rows stop ranking. A candidate with fewer scored observations than `scoring.minimum_seed_observations` enters the unranked receipt with its reason.

## Cofold Confidence

| Measurement | Definition | Use |
|---|---|---|
| `ipsae_target_to_binder` | Directed interface predicted aligned error score from target to binder | Raw predictor measurement |
| `ipsae_binder_to_target` | Directed interface predicted aligned error score from binder to target | Raw predictor measurement |
| `ipsae_min` | Minimum of the two directed ipSAE values | Primary ranking measurement |
| `iptm` | Complex or configured chain-pair interface TM confidence | Supporting confidence |
| `interface_pae` | Mean or registered interface PAE definition | Supporting confidence and threshold |
| `interface_plddt` | Registered interface confidence on a 0–100 scale | Supporting confidence and threshold |

Each observation records the target and binder chain IDs, chain mapping, ipSAE implementation revision, interface cutoff, model revision, predicted-complex path and hash, PAE path and hash, metric-source path and hash, and raw prediction-record hash. Every target-predictor cohort must use the campaign’s target hash, model revision, metric implementation, cutoff, and site-scoring contract. The ranker checks that `ipsae_min` equals the minimum of the two directed values. Terminal validation proves that the normalized measurements refer to an emitted raw predictor row and that every numerical value matches its hashed metric-source record. Stage receipt validation also rehashes the referenced complex, PAE, and metric-source files.

## Pose Consistency

`sc_dockq` compares the independently predicted complex with the generator’s design pose after target-chain alignment. The observation records both paths and hashes, the DockQ implementation revision, chain mapping, aligned target-residue count, target-alignment RMSD, mapping result, DockQ, Fnat, interface RMSD, and ligand RMSD.

`scoring.seed_aggregation` selects the reducer used by promotion and final ranking. Custom campaigns may choose `max`, `min`, `mean`, or `median`. The extrema select a row by the configured primary metric and retain its paired pose metric; ties select the lowest seed. Published baseline fidelity uses maximum `ipsae_min` with paired `sc_dockq`. Reports include an alternate reduction from the same observations.

For the published five-seed maximum, 1.163 standard deviations is a theoretical upward bias for five independent normal draws. It is not a measurement of this campaign or a correction for another seed count or reducer.

## Target-Site Contacts

| Measurement | Meaning |
|---|---|
| `site_contact_iou` | Intersection over union between candidate target contacts and the registered site contacts |
| `target_contact_recall` | Fraction of registered site contacts recovered by the candidate |
| `target_contact_precision` | Fraction of candidate target contacts that fall inside the registered site |
| `hotspot_recovery` | Fraction of registered hotspot residues contacted |
| `offsite_contact_fraction` | Fraction of candidate target contacts outside the registered site |

The target contract selects one site mode: explicit residues, reference-partner contacts, reference-pose contacts, or a spatial pocket. Every observation stores the site scorer revision, atom selection, contact cutoff, residue-map hash, and measurement basis.

A target can add a contact-chemistry metric, alternate state, homolog, or antitarget through another scorer adapter.

## Registered Custom Measurements

`scoring.metric_registry` adds target-specific measurements without changing the common observation schema. Each entry declares a metric ID, units, minimum, maximum, reducer, and implementation revision. The metric source records the per-prediction value and revision. `custom_metric_gates` applies candidate thresholds; `custom_control_metric_gates` applies independent positive and negative thresholds.

## Interface Geometry

The common observation includes clash count and contact count. An adapter can add buried surface area, shape complementarity, hydrogen-bond count, salt-bridge count, unsatisfied polar count, hydrophobic burial, or other registered measurements. Each added metric declares its units, implementation, revision, and threshold.

## Filters

The filter stages record one result per candidate and check:

- sequence composition;
- exact duplicate sequences;
- liability chemistry;
- target-chain window sequence novelty;
- model likelihood from the profile-declared candidate metric source;
- structural novelty when a TM-align or Foldseek route is selected and configured;
- secondary structure when a DSSP route is selected and configured.

The template leaves those last two filters skipped. Claude Science can configure their tools when the experiment calls for them.

The campaign pins each active filter ID, stage, metric, operator, threshold, tool revision, reference revision, and reference hash. The filter table stores those fields with the observed value, pass state, and reason. The filter report marks each gate as ran and passed, ran and failed, or skipped with its reason. The screen gate requires exactly one row for every candidate and configured filter, recalculates each threshold result, and derives the passing manifest from the complete matrix.

Every sequence-changing optimization round repeats the active filter matrix. Round predictors consume the derived passing manifest. Final selection joins each optimized child to its round filter rows and preserves its FASTA, canonical sequence hash, sequence length, and design pose.

## Control Separation

Positive and negative controls use the registered target construct, predictor revisions, seed policy, chain mapping, parsers, and measurements. Control IDs, roles, structure hashes, predictor IDs, model revisions, and seeds must match the configured control matrix exactly.

The final control gate runs per predictor and across the ensemble:

- each predictor must recover every configured positive above its ipSAEmin, sc-DockQ, site-IoU, and site-recall thresholds;
- each predictor must satisfy every configured per-control minimum or maximum gate;
- every control must have the complete rescore seed matrix;
- site and contact-chemistry controls can add target-specific separation rules.

The supported control roles include known same-site complexes, matched wrong pairs, sequence decoys, pose decoys, and alternate-site complexes.

Failed predictor calls remain explicit matrix rows with the target, candidate or control, predictor, phase, seed, failure code, and failure reason. Scored rows require the complex, PAE, metric source, mappings, and measurements. A failed row cannot satisfy coverage or enter normalization.

## Ensemble Score

For candidate `c`, predictor `p`, and seed `s`, `max` uses:

```text
s*(c,p) = argmax over registered seeds of ipSAEmin(c,p,s)
```

`min` selects the lowest configured primary metric with its paired pose metric. `mean` and `median` reduce each metric across the observed scored seed values. The rank output carries the observed seed count, mean, median, sample standard deviation, minimum, maximum, and range for every metric in every predictor arm.

For each predictor, calculate population z-scores across rankable, filter-passing candidate rows:

```text
z_ipSAE(c,p) = z-score of ipSAEmin(c,p,s*) within predictor p
z_pose(c,p)  = z-score of sc-DockQ(c,p,s*) within predictor p
```

The published ranking combines the normalized values:

```text
rank_score(c) = mean over predictors p of
                [0.8 × z_ipSAE(c,p) + 0.2 × z_pose(c,p)]
```

For a custom weighted score, select `scoring.ranking_mode: custom-weighted-zscore` and set the primary metric, pose metric, positive relative weights, and `scoring.metric_directions`. A minimized metric contributes its negative z-score. The score divides by the total weight magnitude and averages across the configured predictors. Candidate raw-mean modes retain their named ipSAEmin/sc-DockQ statistic.

Normalization uses rankable, filter-passing candidates. The rank output records the exact candidate IDs and a cohort hash. The final rank also records the propagated rank-score spread. Adjacent rows with a rank-score gap smaller than their pooled spread are marked `tied_within_noise`. The top-ten summary records separable and tied positions.

## Eligibility And Portfolio Selection

A candidate is eligible after it passes:

- the configured minimum scored observations for every predictor arm;
- sequence and structure filters;
- minimum ipSAEmin ensemble;
- minimum sc-DockQ ensemble;
- maximum clash count;
- all configured counter-target or state-specific gates.

Site-contact IoU and target-contact recall are diagnostics and ranking tie-breakers. `final-rank` does not apply their configured threshold fields as eligibility gates. `rank_score` has no fixed threshold. The campaign config selects the ipSAEmin, sc-DockQ, and clash-count eligibility thresholds under `scoring.thresholds`.

Eligible candidates sort by rank score and the configured tie breakers. Tie order makes output reproducible and carries no evidence of superiority. Under `max` or `min`, compared candidates must have the same scored fold count for every predictor arm. Unranked candidates retain their reason in the ranking receipt. Portfolio selection reserves the required origin generators, then fills remaining positions in rank order within the configured caps.

Screen promotion and optimization-round selection use the same principle at smaller scope. The executor recomputes coverage, controls, registered gates, deterministic order, count caps, and generator constraints from the upstream measurement cohort. A tool-emitted promotion or eligible-parent file passes only when it matches that recomputation exactly.

Selection defaults follow the enabled campaign methods. The minimum Levenshtein distance remains 6. The maximum root-backbone fraction remains 0.05. The maximum TM90 cluster fraction remains 0.10. The maximum structure-method fraction is 1.0 with one enabled structure generator and 0.50 with two or more. The maximum sequence-method fraction is 1.0 with one enabled sequence designer and 2/3 with two or more. The minimum structure-method count is the smaller of 3 and the enabled generator count. A campaign with no enabled generators remains invalid.

Promotion can deliver fewer parents when diversity or the eligible pool binds. `selection.shortfall_policy.minimum_delivery_fraction` controls the lowest accepted delivered fraction. Its default is 0.5. The promotion summary records the requested count, delivered count, and binding rule. The delivered count is used by final ranking and output validation.
