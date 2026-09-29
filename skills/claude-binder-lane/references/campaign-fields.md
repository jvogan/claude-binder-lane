# Campaign fields

Use this reference when Claude Science edits a Binder campaign. A scientist
normally supplies a scientific request, not a JSON document. Claude Science
starts from the selected profile and campaign template, resolves what it can,
then changes the fields that express the chosen experiment. The package schema
and `lane check` are the authority for the complete resolved configuration.

Do not fill every optional field or copy a generic configuration skeleton. A
profile supplies defaults for its own route and workflow. A custom workflow may
omit profile-specific machinery that it does not use.

## Normal configuration flow

1. Choose or create a campaign record and an execution profile that match the
   desired target, tools, and route.
2. Compose them, then inspect the resolved configuration and `lane check`
   result. Resolve only errors for the route and claim that the campaign selects.
3. Materialize a plan before any paid work. The plan derives stage expansion,
   workload, and the records that later execution needs.
4. Preflight and persist approval only when the plan includes paid stages.

Claude Science performs these operations through the skill kernel. A shell or
integration can use the same `compose`, `check`, `materialize`, `preflight`, and
`execute` commands documented in [running a campaign](running-a-campaign.md).

## Fields that express the experiment

| Decision | Main fields | Use |
| --- | --- | --- |
| Target and site | `targets[]`, `binder.binder_chain_id`, `binder.target_chain_id` | Identify the input structure, source, design-target chain, residue map, and site rationale. A scientific target run needs a real structure, chain, and site. |
| Reproduction claim | `profile.profile_id`, `profile.baseline_fidelity`, `profile.published_designer_deviations` | Compare the selected configuration with the published protocol. The shipped reference profile discloses its Genie3 C-alpha ProteinMPNN substitution and sets `baseline_fidelity: false`. A reproduction claim needs the resolved record and run evidence. |
| Tools and route | `generation.generators[]`, `sequence_design.designers[]`, `cofold.predictors[]`, `provider` | Select roles, adapters, provider route, credentials, and artifact return behavior for the Binder stages that use them. A native or provider route outside Binder records the same handoff facts outside these fields. |
| Use terms | `declared_use` | State commercial or non-commercial use when the selected tools require the licence gate to evaluate it. |
| Scale and length | `generation.generators[].backbone_count`, `sequence_design.sequences_per_backbone`, `selection.final_count`, `binder.minimum_length`, `binder.maximum_length` | Set the requested candidate volume and binder range. Binder accepts any positive ordered range. Check the selected tool's actual length limits before dispatch. |
| Replication | `cofold.screen_seeds`, `cofold.rescore_seeds`, `scoring.seed_aggregation` | Set the screen and rescore observations plus the reducer used to select the best observation. A custom campaign can use an appropriate nonempty unique seed set and `max`, `mean`, `median`, or `min` reduction. Published baseline fidelity keeps its declared seed policy. |
| Scoring and controls | `controls`, `filters`, `scoring.primary_metric`, `scoring.pose_metric`, `scoring.rank_weights`, `scoring.ranking_mode`, `scoring.metric_directions`, `scoring.thresholds` | Define exploratory ranking or a target-specific winner rule. Use `custom-weighted-zscore` to preserve a chosen objective, pose metric, and weights across any supported predictor set. Set a metric direction when lower values are better. |
| Spend | `provider.budget.maximum_spend_usd`, `provider.budget.currency` | Set a positive USD ceiling only for a paid Binder plan. The plan and approval record provide the estimated spend. |
| Optimization | `optimization.enabled`, `optimization.rounds`, `optimization.parent_count_per_round`, `optimization.variants_per_parent`, `optimization.operators`, `optimization.early_stop_margin` | Choose whether to run the optimizer and how many rounds to budget. A numeric stopping margin permits early stopping; `null` runs every configured round. |
| Output | `viewer_renderer` | Select an optional renderer. The materialized plan and receipts identify the actual outputs. |

`optimization.enabled` controls whether Binder materializes optimization rounds.
It may be `false` for a custom or non-baseline workflow; a baseline-fidelity
campaign keeps it `true` because the published protocol includes optimization.
`optimization.rounds` is the fixed round budget when optimization is enabled.
`optimization.early_stop_margin` may be a finite nonnegative number or `null`.
`null` disables early stopping, so the configured round budget runs in full. A
numeric margin allows an early stop only when its recorded or declared metric
basis matches the selected objective. `scoring.ranking_mode` and
`scoring.seed_aggregation` control the resulting rank and promotion path; do not
replace the configured mode or reducer because a profile uses a different
default.

## Descriptive metadata

Some generator entries carry `standalone_without_designer`. This field describes
a profile whose generator also produces sequences. The executor does not read
the flag; the profile's declared stages determine whether a separate sequence
designer runs. To change execution, edit the stage list.

## Claims and validation

`lane check` validates configuration shape, route bindings, selected licence
terms, target files, stage contracts, and claim-specific requirements. A valid
configuration does not prove a provider ran, a model revision was available, or
a candidate binds its target. Use receipts and scored outputs to support those
separate statements.

For exploratory work, report the score rows, configuration, and uncertainty.
For a target-specific winner claim, provide the configured controls,
thresholds, scoring provenance, and replication appropriate to that claim.
[Target qualification](target-qualification.md) explains that distinction.

## Stored control separation

For a target-specific scoring claim, store the measured positive and negative
control separation under 'qualification.control_separation'. Record the target,
scoring arm, statistic, source observations, and threshold that the
qualification run supports. Use a target-specific
'qualification.control_separation.thresholds' entry when one arm needs its own
threshold; use 'qualification.control_separation.threshold' only when the same
threshold applies to every recorded pair.

The stored measurement records evidence. It does not create a threshold for a
new target, model revision, route, or seed policy. Read
[target qualification](target-qualification.md) before using it in a winner
claim.

## Paid plan approval

After materialization, `run_intent.py` creates the plan-bound approval record.
It needs the materialized plan, a rate record for the selected route, the plan
freeze digest, and `<bundle>/approvals.jsonl`. The paid executor reads that
record before dispatch. [Approval and spend](approval-and-spend.md) explains
what the approval must cover.

Do not create an approval ledger for local planning, a free native route, or the
optional local diagnostic. The freeze digest is a machine record for this exact
plan. Claude Science refreshes it from an active authorization when a revised
plan remains within the authorized provider, data, hardware, workload, and
ceiling. Ask the scientist again only when a revision exceeds that scope or the
authorization expires.

## When validation fails

Read the error before changing configuration. Resolve a missing target file,
provider endpoint, model revision, or account setting from the selected route.
If a Binder adapter is unbound, use a bound adapter, a native Claude Science
route, or a provider-native route and preserve its artifacts for the next stage.
Do not invent values merely to make a template compose. [Troubleshooting](troubleshooting.md)
maps common failures to the relevant reference.
