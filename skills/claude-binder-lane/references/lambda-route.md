# Lambda Cloud route

Binder ships a generic guarded lifecycle for Lambda Cloud. It does not ship a
Lambda Cloud HTTP transport, authentication scheme, account read, launch call,
status mapping, terminate call, worker image, or a recorded Lambda run. A
Lambda profile or route-matrix row is a configuration template. It is not an
executable or verified Lambda route.

Use Lambda Cloud's own console, CLI, API, or a Claude Science native route when
you want to run a Lambda image today. Preserve the selected image and model
revisions, request parameters, returned artifacts, route, and cost evidence.
Bring those artifacts into Binder at the next stage when they satisfy its input
contract. [Connector authoring](connector-authoring.md) describes how to make
that handoff reusable.

## What Binder ships

`lambda_platform.py` reuses the guarded lifecycle that Binder uses for native
cloud work: approval, spend reservation, durable job identity, no-resubmit
resume, receipt validation, cleanup, and financial reconciliation.

`lambda_client.py` supplies the interface and safety checks for a transport.
`LambdaTransport` accepts three caller-provided functions:

| Function | Required result |
| --- | --- |
| `launch` | A durable provider job ID. |
| `status` | Binder's `pending`, `succeeded`, `failed`, `cancelled`, or `timed_out` state, plus an observed exit code and usage when available. |
| `terminate` | A successful cleanup result, including after the job has stopped. |

The package has no Lambda Cloud facts that implement those functions. Its HTTP
helper can validate a caller-supplied HTTPS URL and Authorization header, but it
does not know a Lambda endpoint path or authentication format. It cannot
authenticate or launch work by itself.

## What Binder execution needs

Claude Science can use the provider's documented API to implement and bind a
transport, worker, and artifact handoff within the authorized scope. It presents
the deployment and run estimate before it creates paid infrastructure. Until a
parsed result exercises that work, report it as configured or unverified rather
than as a ready Binder route. Binder execution needs:

1. The provider's documented authentication header and a session credential
   variable name.
2. Actual launch, status, and termination calls, including the translation from
   Lambda status values to Binder's lifecycle states.
3. A read-only account call that verifies the credential without allocating
   compute.
4. The selected instance type, region, image, model revisions, storage,
   artifact return behavior, rate source, data authorization, and budget.
5. A worker that implements Binder's `intent`, `command`, `inputs`,
   `outputs`, and `run_timeout_s` request contract, then returns an observed
   exit code and declared artifacts including `receipt.json`.
6. A parsed N=1 run that exercises the exact image, transport, artifact
   retrieval, receipt, and downstream parser.

`instance_type_name` and `region_name` in the template are operator values,
not verified Lambda API field names. A completed direct Lambda run can establish
provider facts, but it does not make the Binder transport verified until the
same transport and worker have been exercised.

## Binder execution after bring-up

Bind the verified host and read-only account reader in the Claude Science
kernel. The ordinary Binder execution path then records `submission-intent`,
reserves the approved maximum, and waits for a durable job ID. If provider
acceptance is ambiguous, Binder preserves the reservation and requires the
operator to reconcile the job ID before resume. It never submits a second job
for the same unresolved intent.

The transport must harvest the declared outputs and `receipt.json`. Binder
validates both before success and requests cleanup for terminal, timeout, or
failed-harvest paths. Missing provider usage leaves the financial record
`pending-provider-usage`; Binder does not invent a settled amount.

## Route status

The lifecycle interface is shipped. Lambda Cloud transport and worker
qualification are not. This limits only automated Binder execution on Lambda.
It does not limit a direct Lambda workflow or an artifact handoff.

Source: [`lambda_client.py`](../claude_binder/clients/lambda_client.py)
defines the transport interface. [`lambda_platform.py`](../claude_binder/adapters/lambda_platform.py)
forwards the lifecycle to [`runpod_platform.py`](../claude_binder/adapters/runpod_platform.py).
