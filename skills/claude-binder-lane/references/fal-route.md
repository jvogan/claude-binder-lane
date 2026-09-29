# Use fal from Claude Science

fal offers existing model endpoints and user-deployed Serverless apps. Dedicated
Compute is a separate route with its own account access. Select the surface
that fits the requested tool. A missing marketplace endpoint leaves private
deployment available to assess.

The [official Serverless guide](https://fal.ai/docs/documentation/serverless)
describes custom containers, private deployment, and model loading during setup.

## Resolve the route

Use the session's fal skill and read-only account inventory to find an existing
app before asking for a deployment URL. Check its owner, access policy, tool
revision, and request contract. Keep private URLs in the campaign's resolved
configuration. Catalogue placeholders preserve portability; they are tasks for
the agent to resolve.

If no suitable app exists, prepare a private app definition and an estimate for
the user's account. Deployment and execution require the authorization described
in [approval and spend](approval-and-spend.md). API credentials used to invoke an
app may lack permission to deploy it. Name that missing permission if discovery
confirms it. Reuse recorded rates for a familiar machine and workload; refresh
them when the selected shape or billing evidence changes.

## BindCraft2 on private compute

BindCraft2's pinned licence permits private-cloud use and user-installed copies
directed by the user's own tooling. It restricts providing the software's
functionality to third parties as a hosted service. Apply those conditions to
the actual installation and access arrangement on either fal or Modal.
An endpoint in a user's account requires private access and an installation for
that user's own use. Read the [licence record](tool-licences.md) for the full terms.

The shipped BindCraft2 adapter launches a process inside its worker. It accepts
an executable path, and has no HTTP client. Run that adapter inside the chosen
private worker, or use the provider's native interface and retain the output
contract. Endpoint flags belong to the transport around the worker. Their
rejection by this process adapter does not prohibit the provider.

The [recorded BindCraft2 route](current-status.md#bindcraft2-on-modal-2026-09-21)
uses Modal. A fal deployment still needs implementation and a bounded first run
before claiming that route has been verified. Preserve the user's provider
choice and prepare that work when requested.

## Credentials and calls

In Claude Science, declare the stored `FAL_KEY` credential on the cell that
needs it. An undeclared cell can see an empty environment even when the
credential exists. Check presence without printing its value. This uses the
packaged helper without launching a request:

```python
from claude_binder.clients.fal_invocation import credential_present, resolve_route

credential_available = credential_present()
route = resolve_route("fal-credential-wrapper")
```

Automatic routing uses a declared credential directly, then falls back to the
configured workstation wrapper. Read the installed helper signature before
constructing a custom invocation. The packaged clients already use that helper.

A health check or toolcheck can start a billed worker. Use account metadata for
free discovery and include any worker call in the approved estimate. Preserve
request IDs and reconcile remote state before retrying a client timeout.
Keep private artifacts in authenticated storage. The
[fal cost record](measured-costs.md#fal-esmfold2-fast-the-per-request-model-load-is-the-cost)
distinguishes historical application costs from provider-wide claims.
