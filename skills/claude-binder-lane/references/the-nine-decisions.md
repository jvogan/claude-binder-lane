# The nine decisions

These are the choices that turn a binder-design question into an executable
campaign and an honest result. They are not an onboarding checklist. Start with
the request, settle the relevant choices together, and leave a decision open
until its evidence or authorization exists.

| Decision | What it settles |
| --- | --- |
| 1. Target and site | The structure, construct, target chain, and design surface. |
| 2. Reproduce or substitute | Exact published-stack reproduction, a named swap, or a custom workflow. |
| 3. Compute route | Local, platform-managed, hosted API, or self-hosted execution for each stage. |
| 4. Tool stack | Generator, designer, co-folder, scorer, and the actual route for each. |
| 5. Scale and replication | Candidate volume, length range, seeds, and delivery count. |
| 6. Winner rules | Objective, controls, thresholds, and the interpretation of a ranking. |
| 7. Spend ceiling | Provider scope, maximum cost, and plan-bound approval. |
| 8. Rounds and stopping | Optimization work, its objective, and its stopping policy. |
| 9. Evidence and handoff | Inputs, receipts, results, retention, and reportable claims. |

## 1. Target and site

Choose a structure, source identity, target chain, and site rationale. A
partner complex can supply the site through an explicit contact rule. A supplied
residue list can also define it. Claude Science can inspect public structures
and prepare residue maps, but it cannot invent a site rationale.

If the target or site is missing, Claude Science can still discover tools,
compare published targets, and propose a plan. It cannot dispatch a
target-specific scientific campaign until the structure, chain, and site are
defined. [Target inputs](target-inputs.md) and [published
targets](published-targets.md) provide the needed evidence.

The Binder record uses 'targets[]' and the 'binder' chain fields. See
[campaign fields](campaign-fields.md) only when editing the record.

## 2. Reproduce or substitute

Choose whether the goal is exact reproduction of the Anthropic computational
stack, a stated substitution, or a custom workflow. Exact reproduction selects
the published roster and its declared seed, scoring, control, and ranking
policy. It is not a claim that every selected endpoint, checkpoint, or route is
already configured or verified.

Use [reproduction readiness](reproduction-readiness.md) to distinguish four
facts: published capability, Binder binding, selected-profile configuration, and
recorded execution. 'published-baseline-fidelity.template.json' expresses the
published-stack selection with a disclosed Genie3 designer substitution:
ProteinMPNN C-alpha mode handles its C-alpha trace. The profile does not claim
exact fidelity. A credential-specific
checkpoint denial applies to that credential; it does not remove the route from
planning.

For a substitution, name the replaced tool, route, revision, and changed
interpretation. Preserve the published ranking only when the selected
configuration still meets its stated requirements. Otherwise choose and report
the intended 'scoring.ranking_mode'. A custom weighted ranking can preserve the
selected primary metric, pose metric, and weights across a supported predictor
set. [Published campaign comparison](published-campaign-comparison.md) records
the protocol and claim limits.

## 3. Compute route

Choose the execution surface for each selected stage: local process, Claude
Science platform skill or managed endpoint, hosted API, or self-hosted Modal,
RunPod, Lambda, or fal deployment. Mixed routes are normal when they improve
the tool fit, evidence, or cost.

The [tool catalogue](tool-catalogue.md) and [platform tool
discovery](platform-tool-discovery.md) describe known routes. They are not an
allowlist. An unbound Binder adapter prevents only a Binder-executed stage. A
Claude Science native skill or provider-native route can still run after its
real setup and authorization. Preserve the returned artifacts, model identity,
parameters, and route evidence for downstream work.

Claude Science resolves available platform skills, provider inventory, and
provider-native setup where it can. It asks before a route sends data to a new
host or starts a paid provider job.

## 4. Tool stack

Choose tools by scientific role, then choose the route for each selected tool.
Use the catalogue to compare capabilities, model lineage, published evidence,
licence terms, route requirements, and known limitations. A missing package
adapter never proves that a tool is unavailable.

State 'declared_use' when the selected tools require a commercial-use decision.
Check the actual code, weights, deployment, and license terms for the selected
use. Resolve a published tool's endpoint, model revision, image, or credential
from its own documentation or the platform inventory. If a route remains
unconfigured, present it as a configuration gap and propose a disclosed route
that serves the same scientific aim. Do not silently swap tools.

Binder records its selected stages in 'generation', 'sequence_design', 'cofold',
and 'provider'. Read [connector authoring](connector-authoring.md) when a native
or provider route needs a reusable Binder handoff.

## 5. Scale and replication

Choose candidate count, sequences per backbone, desired delivery count, binder
length range, screen observations, and rescore observations. Start at the scale
the question needs. Do not reduce user-requested scale or change a requested
range without stating the reason and effect.

Binder accepts any positive ordered 'binder.minimum_length' and
'binder.maximum_length'. The selected tool may have tighter real limits, which
the agent checks before execution. Custom campaigns can use a suitable nonempty
set of unique rescore seeds and a 'max', 'mean', 'median', or 'min' reducer.
Published baseline fidelity uses the declared published seed policy.

The materialized plan derives workload and cost from the selected route and
scale. [Measured costs](measured-costs.md) distinguishes recorded charges,
provider rates, and planning estimates.

## 6. Winner rules

Choose the primary objective, pose metric, metric directions, controls,
thresholds, and ranking policy. An exploratory campaign can produce a ranked
table without a target-specific winner claim. A winner claim needs controls,
provenance, replication, and thresholds that support that claim on this target.

For a custom objective, set its direction when lower values are better and
select a ranking mode that preserves the stated objective and weights.
'custom-weighted-zscore' keeps user-selected metric weights rather than
replacing them with a profile default. The configured seed reducer controls
which observation promotion and final ranking use.

[Target qualification](target-qualification.md) and
[scoring details](scoring-details.md) explain the evidence a result needs.

## 7. Spend ceiling

Before a paid Binder stage, choose the provider scope and a positive maximum
spend in USD. Claude Science materializes the plan, evaluates the selected
route and scale, and presents the provider, hardware, data destination,
estimate, and effective ceiling. An account-policy ceiling applies when it is
lower.

The approval record binds those terms to the plan. An existing authorization
continues to cover a revised plan within its provider, route, hardware, data,
workload, and ceiling scope; Claude Science refreshes the plan record. It asks
again only when a revision exceeds that scope or the authorization expires. Free
planning and locally executed work require no spend approval.
Read [approval and spend](approval-and-spend.md) for the execution boundary.

## 8. Rounds and stopping

Choose whether optimization is enabled, how many rounds to run, the parent and
variant counts, operators, objective, and stopping basis. The
'optimization.enabled' field is authoritative. A profile label cannot silently
disable requested optimization.

The campaign must name the metric that the optimizer improves and the basis of
any early-stop margin. Claude Science does not add optimization rounds unless
the selected policy authorizes them. It also does not remove requested rounds or
change their objective to satisfy a profile default. Record the resulting
continue or stop decision with the round summary and score table.

## 9. Evidence and handoff

Retain the target inputs, site record, selected tool and model revisions, route,
parameters, materialized plan, approval, receipts, score records, output
artifacts, and hashes needed for the result you will report. A computational
ranking supports a prioritized design list. It does not prove binding, affinity,
specificity, or biological activity.

Use [keeping results](keeping-results.md) for durable artifact handling and
[resume](resume.md) for an interrupted Binder run. An unavailable persistence
destination limits only that delivery path; preserve the local or
provider-returned artifacts while a durable handoff is arranged.

## Configuration reference

Use [campaign fields](campaign-fields.md) for the fields that express these
decisions. Let the selected profile and 'lane check' supply the complete
route-specific configuration. Do not add unused fields, invent values to clear
placeholders, or turn an optional diagnostic into a campaign requirement.
