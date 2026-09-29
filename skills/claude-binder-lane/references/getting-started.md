# Start a Binder campaign in Claude Science

Claude Science operates this skill with you. Start by describing the target,
site, desired result, and any preferences for tools, providers, budget, or
reproducing the published stack. You do not need to load Python, find the skill
directory, run a fixture, or translate your request into a campaign file.

The skill kernel loads with the skill. Claude Science uses it to inspect the
installed package, discover routes, prepare inputs, compose a Binder plan when
appropriate, and retain results. If the kernel did not load, reopen the skill.
Read [package loading](package-loading.md) only when that recovery fails.

For a bounded first study, give Claude Science the request in [small campaign
fast path](small-campaign-fast-path.md). It starts with one candidate through
the selected handoffs and a target-matched control panel, then scales inside
the agreed ceiling. A native tool route remains available when the complete
Binder graph is unnecessary or has not been verified for the chosen tools.

## What Claude Science does before it asks you to act

Claude Science reads the selected profile, the catalogue, available platform
skills, provider inventory, and free route checks. It proposes a campaign with
the selected tools and routes, what each route needs, expected scale and cost,
and the scientific claim the run could support.

When you leave the provider open, Claude Science selects a suitable connected
route and explains why. It resolves executable paths, prepares image definitions,
and records model and environment identities from the chosen installation.
These are setup tasks for Claude Science. It asks you about access only after
checking the session's existing connections and credentials.

Claude Science verifies installation values against the image and receipts.
It records checkpoint digests after setup instead of asking you to invent a
model identity or confirm paths it can inspect. Preparing a runbook supports
execution; a run request continues through the authorized work.

For BindCraft2, connected Modal is the first route to assess because a packaged
image recipe and a recorded tool run already exist. Use the
[BindCraft2 setup](tool-catalogue-details.md#bindcraft2-bound-on-2026-09-21)
to prepare the selected image and carry one design through its downstream
screen. A request for another provider takes precedence.

It can use a native Claude Science skill or a tool's hosted API even when this
package has no adapter for it. In that case, it preserves the source artifacts,
tool revision, parameters, and output identity for the next stage. An unbound
Binder adapter is not a reason to hide a native route.

Claude Science asks you only for facts it cannot safely infer or obtain:

- The target, construct, site rationale, and any data-use restriction.
- A requested reproduction claim or a deliberate substitution.
- A paid provider, data egress, and a maximum campaign spend.
- Target-specific controls, thresholds, or interpretation when you want a
  winner claim rather than an exploratory ranking.

The [decision reference](the-nine-decisions.md) helps with these choices when
they arise. It does not impose an order or require every field before planning.

## When a route needs access

Local planning and the optional diagnostic need no provider account. Connect a
provider only when a selected stage uses that provider. A hosted API or a
self-hosted job can require a credential, data egress permission, and provider
billing; Claude Science identifies the host and purpose before making that
request.

For Modal, connect the account in **Settings** > **Compute** before a selected
Modal job. For other providers, use the account and credential route documented
by that provider. [Provider authorization](provider-authorization.md) has
provider-specific recovery details.

## Before paid execution

Claude Science materializes and preflights the chosen plan. It presents the
provider, route, hardware, data destination, scale, estimate, and effective
ceiling. The approval binds that plan. A previous authorization remains usable
when revised terms remain within its scope, and Claude Science refreshes the
plan-bound record. It asks again only when the provider, hardware, data egress,
scale, or ceiling exceeds the authorized scope or the authorization expires.

The plan requires a positive USD ceiling and a provider-backed estimate before
it can dispatch paid stages. This is a real spend boundary, not an onboarding
task. [Approval and spend](approval-and-spend.md) explains the recorded
approval and account-policy cap.

## Results and persistence

Claude Science saves the materialized plan, resolved configuration, inputs,
receipts, score records, and chosen outputs when the run produces artifacts.
Those records support later inspection and an honest description of the route.
Use [keeping results](keeping-results.md) when you need a particular retention
destination or a downloadable handoff.

After the first output passes the planned checks, Claude Science continues
within the approved scope. It pauses when you requested a review point or when
the result requires a new scientific, data-use, or budget decision.

An optional [local diagnostic](free-run.md) exercises the package without
network or money. It is useful after installation trouble or a package change.
It is not a first-run requirement and does not validate a scientific target,
tool, or result.
