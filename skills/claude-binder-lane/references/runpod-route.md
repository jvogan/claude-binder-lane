# RunPod Serverless route

Binder ships a RunPod Serverless HTTP client and a guarded job lifecycle. It
does not ship a deployed endpoint, a Binder worker image, or evidence that a
worker implements Binder's request and artifact contract. A profile or
route-matrix row that names RunPod describes a selectable configuration shape,
not a ready or verified execution route.

Use RunPod's own console, CLI, API, or a Claude Science native route when that
is the practical way to run your image. That work remains first class. Record
the image and model revisions, request parameters, output artifacts, route, and
cost evidence. Import the artifacts at the next Binder boundary when they meet
the downstream contract. Read [connector authoring](connector-authoring.md) if
you want to automate that handoff.

## What Binder ships

`runpod_client.py` calls documented RunPod Serverless endpoints to:

- Read the account's endpoint inventory.
- Read endpoint health.
- Submit, poll, and cancel a job for a configured endpoint ID.

`runpod_platform.py` supplies the Binder lifecycle around that client:
plan-bound approval, read-only authorization, spend reservation, durable job
identity, no-resubmit resume, receipt validation, cleanup, and later financial
reconciliation. These components use no additional runtime skill or plugin.

The client sends a rendered stage command as data to your worker. RunPod does
not execute that command by itself. Binder's worker request has `intent`,
`command`, `inputs`, `outputs`, and `run_timeout_s`. The worker must
return an integer `exit_code` and make the declared artifacts and
`receipt.json` available to the harvest step.

## What Binder execution needs

Claude Science resolves available account facts and can prepare or adapt the
selected worker and artifact handoff within the authorized scope. It presents
the deployment and run estimate before it creates paid infrastructure. The
following route state is needed before Binder can dispatch:

1. A Serverless endpoint on your account, with its ID in
   `adapter.runpod.endpoint_id`. The shipped client refuses a missing ID.
2. An image and worker that implement the Binder request, output, and receipt
   contract. The package has not deployed or qualified one.
3. A credential variable, normally `RUNPOD_API_KEY`, and a read-only account
   binding. The account read verifies access. It does not verify the worker.
4. The selected image, model pins, storage and artifact return contract, rate
   record, budget ceiling, and data authorization.
5. A parsed N=1 result from this endpoint and worker before a wider paid
   campaign. Verify the returned artifacts, receipt, and downstream parser.

The template's `endpoint_id: __REQUIRED__` marks a real configuration gap. A
RunPod account or endpoint ID alone does not establish the worker contract.
That is implementation work for Claude Science or an operator, not a reason to
discard the selected RunPod route. Do not copy a Modal or fal receipt, rate, or
image identity into this route.

## Binder execution after bring-up

Bind the authenticated RunPod client host and its read-only account reader in
the Claude Science kernel, then use the ordinary Binder execution path. Binder
records `submission-intent` before it sends the job. If the provider response
does not yield a durable job ID, Binder records `submission-uncertain` and
keeps the spend reservation. Inspect the provider ledger, bind the recovered job
ID, and resume. Do not submit a second job while that intent remains unresolved.

On resume, Binder attaches to the recorded job rather than resubmitting it. It
validates the harvested receipt and artifacts, records the terminal state, and
cancels jobs whose terminal state was not observed. RunPod status provides no
per-job settled USD figure, so a scientifically valid receipt can remain
`pending-provider-usage` until account billing data reconciles the reservation.

## Route status

The HTTP transport is shipped. The Binder worker contract and a provider run
through it are unverified. This is a route-specific limitation. It does not
limit a direct RunPod execution or a different provider route.

Source: [`runpod_client.py`](../claude_binder/clients/runpod_client.py)
implements the Serverless calls. [`runpod_platform.py`](../claude_binder/adapters/runpod_platform.py)
implements the guarded lifecycle. [Provider authorization](provider-authorization.md)
describes the read-only account check.
