# Approval and spend

Approve a paid campaign after Claude Science materializes a plan and shows its
provider, route, hardware, data destination, maximum estimate, and USD ceiling.
The package records that approval against the plan and refuses a paid stage that
does not have one.

Set `provider.budget.maximum_spend_usd` to a positive USD amount for each paid
Binder plan. If an account-policy cap exists, the lower cap applies. The
campaign cannot raise an account-policy cap.

Use a current provider rate record and the plan's workload estimate to choose a
ceiling. Treat an estimate as an estimate: provider billing, cold starts,
idle resources, and manually operated endpoints can differ. [Measured
costs](measured-costs.md) labels settled charges, provider rates, and modeled
figures separately.

Persist the approval with the plan's freeze digest and approval ledger before
dispatch. Claude Science performs this bookkeeping for an authorized campaign.
The command contract is documented in [campaign fields](campaign-fields.md#paid-plan-approval)
for a shell or integration that needs it.

An existing authorization covers later execution while the provider, route,
hardware, data scope, planned workload, and ceiling remain within its recorded
terms. The package refreshes the plan-bound freeze record for that changed plan;
it does not turn routine replanning into another request to the scientist. Ask
again only when a revised plan exceeds the authorized terms or the authorization
expires.

An authorization for a campaign can include its named qualification canaries.
Give each canary a dedicated `--max-cost-usd` ceiling and accounting record.
Those execution records do not require another human confirmation. A canary
outside the authorized provider, data, hardware, workload, or ceiling scope
does.
