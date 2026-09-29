---
name: claude-binder-lane
description: >-
  Plan and operate computational protein-binder design campaigns in Claude
  Science. Use for a published-work reproduction or a custom campaign that
  chooses targets, sites, tools and compute routes, spend ceiling, rounds, scoring,
  and evidence. Consult the catalogue when comparing platform skills, hosted
  APIs, and self-hosted infrastructure.
license: MIT
metadata:
  display-name: Claude Binder Lane
---

# Claude Binder Lane

Use this skill to plan and operate computational binder campaigns in Claude
Science. Work with the scientist to choose the target, site, tools, routes,
reproduction goal, budget, rounds, and objective. Claude Science prepares the
plan, operates selected tools, and reports supported computational claims.

The package provides campaign execution, receipts, scoring, and catalogue data.
It can combine Binder adapters with Claude Science native skills, hosted APIs,
local processes, and self-hosted Modal, RunPod, Lambda, or fal routes. For an
unbound route, use its documented or native interface and preserve the artifacts
and provenance required by the next stage. Use Binder to compose and run a plan
when that adds value. Read [connector authoring](references/connector-authoring.md)
when that handoff needs a reusable Binder bridge.

## Choose the entry point

| Request | Open first | Then read |
| --- | --- | --- |
| One target, BindCraft2, about 24 to 60 designs, and a bounded budget | [Small campaign quickstart](references/small-campaign-fast-path.md) | Only the selected tool and provider references named there. |
| A different generator, supplied candidates, multiple methods, or a close reproduction of the published study | [Full campaign workflow](references/running-a-campaign.md) | The selected profile, stage contracts, and reproduction references. |
| A tool or route comparison | [Tool catalogue](references/tool-catalogue.md) | Only the entries for candidate tools and routes. |

For a BindCraft2 small campaign, begin with the quickstart alone. Collect the target
construct, site or partner-complex rationale, and budget before opening other
references. Its approval and receipt steps apply even when a native Claude
Science tool performs a stage.

Use recorded runs and pinned recipes to choose a starting configuration. Check
the actual provider state, inputs, and model revisions in the scientist's
session. Carry one candidate through a new handoff, inspect its parsed outputs,
and continue within the approved ceiling. A missing prior run for the exact
target or route calls for this qualification step, not a blanket refusal.


## Start with the scientific question

Start with the user's request. Resolve ordinary configuration, installed skills,
provider inventory, and documented tool setup yourself. Use free reads and local
planning commands before asking a question they can answer. Ask for a decision
only when it changes a scientific claim, sends data to a new service, or changes
the authorized bill.

Call `binder_version()` once to identify the loaded build. When
`restart_required` is true, start a fresh conversation before another campaign.
For one requested tool, use `binder_tool_info("bindcraft2")`, substituting its
catalogue ID. Read only that tool's setup and the selected provider reference.

When the user leaves compute open, select a suitable connected provider and
explain the choice briefly. For BindCraft2, prefer connected Modal when it fits:
the package includes an image recipe and records a successful run. Resolve
executable paths, image settings, model identity, and storage yourself. A local
process adapter runs inside a cloud worker too. Read the
[BindCraft2 setup](references/tool-catalogue-details.md#bindcraft2-bound-on-2026-09-21)
before asking where it should run. If the user requests fal, follow the
[fal route](references/fal-route.md).

The shared campaign choices are:

1. Define the target, construct, chain, and design site. Derive the site from a
   supplied partner complex when it fits the question; otherwise request or
   propose an explicit site rationale.
2. Choose exact published-stack reproduction, a named substitution, or a custom
   workflow. Preserve the intended claim in the campaign record.
3. Select tools by role and execution surface. Mix platform skills, hosted APIs,
   local tools, and self-hosted providers as evidence, access, and budget allow.
4. Set scale, replication, controls, scoring objective, winner rule, and rounds
   to answer the question.
5. Before the first paid dispatch, present the provider, data egress, hardware,
   maximum estimate, and campaign ceiling. Persist that approval with the
   materialized plan.
6. Retain input identity, tool and model revisions, route, parameters, receipts,
   score tables, and outputs needed for the reported result.

[The nine decisions](references/the-nine-decisions.md) expands these choices.
It is a decision reference, not a checklist. Read only the sections the request
needs.

## Discover broadly and describe the evidence honestly

Use [tool discovery](references/platform-tool-discovery.md) and the [tool
catalogue](references/tool-catalogue.md) to compare roles, routes, tool terms,
and constraints. The catalogue guides planning; it is not an allowlist. A
missing Binder binding limits only a Binder-executed stage. Native and documented
routes remain available. When a new tool needs a reusable Binder stage, add its
campaign-local catalogue row and connector with [connector authoring](references/connector-authoring.md).

Keep these states distinct in conversation and reports:

| State | Meaning |
| --- | --- |
| Capability | A tool or route is known to exist. |
| Binder binding | This package has an adapter or lifecycle contract for the route. Execution can still need its selected transport, worker, and configuration. |
| Configured | The chosen route has its required account, endpoint, image, or model values. |
| Verified | A recorded run exercised the relevant target, route, and claim. |

Keep the requested tool, scale, model or hardware, and reproduction goal unless
you first state a proposed change and its scientific consequence.

## Reproduce exactly or build deliberately

For an Anthropic-stack reproduction, select
`published-baseline-fidelity.template.json` and compare its resolved
configuration with the published protocol. The shipped profile sets
`baseline_fidelity: false` because its Genie3 C-alpha arm uses ProteinMPNN's
C-alpha checkpoint; SolubleMPNN has no compatible C-alpha checkpoint. The
profile records this designer substitution explicitly. Keep the compatible
route and name the deviation in the report.

Use the full campaign graph for the published selection pattern: generate
across methods, filter candidates, screen each predictor with one seed, advance
survivors to five-seed intermediate scoring, select optimization parents, then
rescore the resulting candidates for final selection. The [published workflow
mapping](references/published-workflow.md) names each stage. The one-generator
[small campaign fast path](references/small-campaign-fast-path.md) runs its own
two-predictor evaluation, with five seeds by default; it does not use this
staged selection graph.
The published-style score takes each predictor's best ipSAE across its seeds
and pairs it with DockQ from that prediction. The small campaign gates on
per-arm mean ipSAE and the maximum pose RMSD across its seeds. Its default
rank averages the two arms' control-normalized mean ipSAE; set
`rescore.ranking_rule: mean_of_arm_best_ipsae` to use the PD-L1 example's
headline score. Keep each formula attached to its route when changing seed
counts, and plan the resulting predictor calls within the approved budget.

[Reproduction readiness](references/reproduction-readiness.md) separates
binding, configuration, and dispatch states. A bound roster can still need an
endpoint, account value, checkpoint, or validation run; report those as gaps and
keep the route visible. Use receipts from the selected configuration and run
before claiming baseline reproduction. A deliberate substitution report names
the substituted tool, route, revision, and changed interpretation. [Published
campaign comparison](references/published-campaign-comparison.md) gives the
protocol and its claim limits.

## Operate the campaign

In Claude Science, load the skill and operate its bound package through the
kernel. The scientist uses conversation to choose the campaign. Claude Science
prepares target inputs, resolves the campaign and profile, preflights a
materialized plan, runs setup and package commands, executes the route, and saves
the artifacts. [Getting started](references/getting-started.md) describes that
session contract. [Running a campaign](references/running-a-campaign.md) has
recovery and execution details when a Binder plan is selected.

For a bounded BindCraft2 study, follow the [small campaign fast
path](references/small-campaign-fast-path.md). For supplied candidates, start
with the [supplied-candidate workflow](references/running-a-campaign.md#start-with-supplied-candidates).
Select the needed tools and carry one candidate through each planned handoff
before scaling. Use a native Claude Science route when it satisfies the
selected stage contract.

Handle transport verification, submission, completion, and artifact retrieval
inside the session. Report progress and results to the scientist. Prepare the
available free setup before presenting a paid-run approval request.

Before generating a batch, check that the selected downstream routes supply the
artifacts required by the planned scores. Use documented schemas and recorded
outputs first. If a handoff remains uncertain, carry the first authorized sample
through that handoff before scaling. Verify chain roles and parse the actual
outputs. [Campaign handoffs](references/running-a-campaign.md#check-handoffs-before-scaling)
covers these checks. Reuse evidence that still matches the selected contract.

The optional [local diagnostic](references/free-run.md) tests packaging and graph
plumbing without contacting a provider. It produces no biological result and is
not required for a scientific campaign.

## Authorization, spend, and evidence

Honor authorization already given for the selected provider, data scope, and
ceiling. Ask again only when the plan changes those terms or the approval has
expired. A paid action requires a real provider account, required data or egress
permission, a positive USD ceiling, and a plan-bound approval record. [Approval
and spend](references/approval-and-spend.md) explains the boundary.

When the selected use triggers tool license or term checks, check the selected
tool, code, weights, deployment, and commercial use. Do not invent a restriction,
price, target, threshold, endpoint, or model revision to clear a configuration
field.

Receipts and provenance must support the claim being made. A computational
ranking supports a prioritized design list; it does not establish binding,
affinity, specificity, or biological activity. Retain the route, versions,
parameters, inputs, score records, and output hashes needed to inspect it. Read
[keeping results](references/keeping-results.md) for durable handoff.

## Reference routing

For a no-spend BindCraft2 plan with one target, start with the
[small campaign fast path](references/small-campaign-fast-path.md). Open at most
three more references: the target input guide if the construct or site needs
resolution, the selected tool's catalogue entry, and the selected provider or
kit recipe. Present the plan from those files. Open the references below while
preparing the approved run or resolving a specific missing fact.

- For target construction, site derivation, controls, and a target-specific result, read
  [target inputs](references/target-inputs.md) and [target
  qualification](references/target-qualification.md).
- For route selection, read [tool discovery](references/platform-tool-discovery.md),
  the [tool catalogue](references/tool-catalogue.md), and the provider reference
  that matches the selected route. A route-matrix row reports a declared profile
  route; read [RunPod](references/runpod-route.md) or [Lambda
  Cloud](references/lambda-route.md) before treating either transport as ready.
  Read [fal](references/fal-route.md) for existing endpoints and private apps.
- For Anthropic's ESMFold2 optimization kit on Modal, read the [pinned image
  recipe and qualification steps](references/esmfold2-kit-modal.md).
- For NVIDIA BioNeMo Inference Runtime as a prediction route, read the
  [BioIR route](references/bioir-route.md).
- For a published-stack request, read [reproduction
  readiness](references/reproduction-readiness.md) and [published campaign
  comparison](references/published-campaign-comparison.md).
- For configuration edits, read [campaign fields](references/campaign-fields.md).
- For price, approval, and a paid run, read [approval and
  spend](references/approval-and-spend.md) and [measured
  costs](references/measured-costs.md).
- For a refusal or failed stage, read [troubleshooting](references/troubleshooting.md)
  and the reference named by the error.
- For measured route-specific evidence, read [current status](references/current-status.md)
  or [new-target provider canary](references/new-target-provider-canary.md).
