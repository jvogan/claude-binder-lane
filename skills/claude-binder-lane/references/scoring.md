# Scoring terms

The package stores computational measurements in observation records. The terms below name the measured fields and the thresholds that gate them:

| Term | Definition | Source |
| --- | --- | --- |
| `ipsae_min` | The smaller of the directed target-to-binder and binder-to-target ipSAE values. | `binder_metrics.compute_ipsae` and the [measurement contract](../claude_binder/data/templates/scoring.md) |
| `sc_dockq` | The pose score calculated by comparing the predicted complex with the design pose after target-chain alignment. | `binder_metrics.compute_dockq` and the [measurement contract](../claude_binder/data/templates/scoring.md) |
| `clash_count` | The number of interchain heavy-atom pairs that the metric implementation classifies as clashes. | `binder_metrics.compute_ipsae` |
| `maximum_clash_count` | The campaign threshold that applies a maximum gate to `clash_count`. | `lane.py` scoring validation |
| `rank_score` | The ranking value for one candidate, calculated by the selected `scoring.ranking_mode`. Candidate raw-mean modes combine the configured primary and pose metrics. `published-three-mode-zscore` uses the published three-mode form. `custom-weighted-zscore` applies the selected primary metric, pose metric, weights, and directions across the enabled predictor set. It is comparable only inside the recorded normalization pool and ranking mode. | `ranking_policy.declared_ranking_mode` and `ranking_policy.rank_score_scope` |

`scoring.primary_metric` and `scoring.pose_metric` record which two metrics the
run ranks on. `scoring.rank_weights` sets their weights. Set
`scoring.metric_directions` to a mapping such as `{"interface_pae": "minimize"}` for a selected metric where lower values are better;
unlisted metrics maximize by default. The published mode retains its published
requirements. Use `custom-weighted-zscore` when a custom objective, direction,
weights, or predictor count requires it.

Each resolved configuration supplies its own scoring thresholds and metric implementation revisions. Do not copy thresholds across unrelated targets.

The gate values in [the gates a candidate must clear](measured-costs.md#the-gates-a-candidate-must-clear) were calibrated on one target's control panel. Another target needs its own controls before those numbers gate it, so most of `scoring.thresholds` ships as `__REQUIRED__` until you fill it. [Scoring details](scoring-details.md) covers control choice, calibration bands, and seed tiers.
