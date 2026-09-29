# Provider authorization

Check authorization after a paid route and materialized plan are selected, before
the first paid dispatch. Modal, fal, RunPod, and Lambda Cloud use separate
read-only bindings. None of these checks starts a campaign stage.

Ask Claude Science for the connection ledger before composing the plan:

```text
compute_details({"provider": "modal", "mode": "read"})
```

Extract the environment `spec_sha` and `im-` image ID. The ledger does not report the workspace name or GPU price. Read workspace identity through the separate kernel call below, and obtain a planning rate from a source that actually reports one. Each environment identity and image reference is scoped to the execution workspace ([Modal bring-up](modal-bring-up.md)).

## Preflight and approval checks

The `preflight` command checks the static contract, the bound read-only Modal identity and environment ledger, and the spend ceiling. The guarded dispatcher's `verify` and submission context separately require the live `host.compute.create` capability and execution callbacks.

| Check | Requirement | Failure action |
| --- | --- | --- |
| `modal-account` | Bound `details_reader` and `identity_reader` values from the current Claude Science connection | Bind closures over the two read results before preflight. If either read cannot authenticate, reconnect Modal in session Settings under Compute. |
| `modal-environment` | Resolved `modal-env:<name>@spec_sha=<hash>` | Read the `spec_sha` from that environment's `compute_details` ledger header and populate `environment_identity`. The authorization gate compares this ledger value; do not substitute the source hash returned by `build_env()`. |
| `image-reference` | Valid image ID matching `^im-[0-9A-Za-z]+$` | Populate `resources.container_image_digest` with the `image_ref` recorded for the same environment in the `compute_details` ledger. |
| `budget-ceiling` | Positive USD amount in `provider.budget.maximum_spend_usd` | Set a positive spending ceiling in the campaign configuration. |
| `modal-authorization` | One free `compute_details` read whose ledger matches every declared environment binding, plus one identity read whose `workspace_name` matches the declared workspace | Bind `details_reader` and `identity_reader` as closures through `configure_binder_modal_platform`, then fix whatever binding the message names. |

## Modal authorization states

`preflight` reports one of four states, and it starts no job in any of them.

| State | Meaning | Effect |
| --- | --- | --- |
| not required | No selected paid stage runs on Modal | Passes, and makes no provider call |
| authorized | The free read succeeded and every declared binding matched the workspace ledger | Passes |
| refused | Conclusive evidence that the declared binding cannot be used | Blocks |
| not measured | No conclusive answer, including an unbound reader, a timeout, or an unreadable response | Blocks |

Both `refused` and `not measured` block. The gate answers whether paid work can proceed, so an unknown answer is not a pass.

An `authorized` result reports that the read succeeded and the declared bindings match. It does not report that the campaign is ready to execute.

Neither read is a Python name. Both are things the agent does outside the
notebook kernel, and a shell `preflight` reaches neither, so it reports
`not measured`. Verified in a live session on 2026-08-30:
`"compute_details" in dir()` returns `False` in the notebook kernel, and
`compute_details` raises `NameError` there. Bind closures over the values the
reads already returned.

Two reads answer two different questions. `compute_details` returns the per-workspace environment ledger:
`### env:<name>@<spec_sha>` blocks carrying `image_ref`, `gpu_default`,
`egress`, `volumes` and `notes`. It never names the workspace and never carries
a price. The workspace name comes only from the identity read.

`token_info` is not a session tool. It is a method on `ModalProvider` in the
platform's `remote-compute-modal/provider.py`, so a session reaches the same
`TokenInfoGet` call through the Modal SDK that the `compute_provider` kernel
already has authenticated. Run this cell there. The first `compute_provider`
cell in a session fires the kernel card once.

```python
# compute_provider kernel
from typing import Any
from modal._utils.async_utils import synchronizer
from modal.client import _Client
from modal_proto import api_pb2

@synchronizer.create_blocking
async def _token_info_get() -> Any:
    client = await _Client.from_env()
    return await client.stub.TokenInfoGet(
        api_pb2.TokenInfoGetRequest(), retry=None, timeout=10.0)

resp = _token_info_get()
print({"workspace_name": str(resp.workspace_name),
       "workspace_id": str(resp.workspace_id)})
```

The call is one unary RPC, costs nothing, and starts no job. A token's
workspace is fixed for its lifetime, so read it once per session.

Then bind both readers in the notebook kernel over the two values you now hold.
Each closure returns a recorded reply and makes no further call, so binding
them costs nothing and repeats no read:

```python
# notebook kernel
import importlib.util, re

details_reply = {"details": """<the details string the compute_details read returned>"""}
identity = {"workspace_name": "<the name the identity cell printed>"}

# Build the capabilities this selected route uses. None is a kernel global.
installed_skills = [<the skill and tool names this session really has>]

def package_resolver(name):
    return importlib.util.find_spec(name) is not None

def _parse_ledger(text):
    # Two entry forms appear in one ledger. Some blocks carry an explicit
    # image_ref line; others put the image ref in the header after @.
    ledger = {}
    for block in re.split(r"^### env:", text, flags=re.M)[1:]:
        head, _, body = block.partition("\n")
        name, _, tag = head.strip().partition("@")
        ref = re.search(r"^image_ref:\s*(\S+)", body, re.M)
        ledger[name] = ref.group(1) if ref else (tag if tag.startswith("im-") else None)
    return ledger

_LEDGER = _parse_ledger(details_reply["details"])

def environment_resolver(environment, image_ref):
    return _LEDGER.get(environment) == image_ref

def render_smoke(stage_id):
    # Return None only after the renderer actually drew a frame.
    # Return the failure text otherwise.
    ...

configure_binder_modal_platform(
    host=host,
    installed_skills=installed_skills,
    environment_resolver=environment_resolver,
    details_reader=lambda request: details_reply,
    identity_reader=lambda: identity,
)
```

`host`, `installed_skills`, and `environment_resolver` are required by the
signature. `package_resolver` is needed only when a selected binding declares
packages. `render_smoke` is needed only when a selected stage produces images.
Pass either optional callback when that route uses it. The call records the
capabilities and invokes none of them. `wait_for_notification` and
`promote_artifacts` remain required before a provider job dispatches.

The preflight checks only selected stages. Missing catalogue tools are warnings
until a selected adapter depends on them. Environment and image resolution,
workspace identity, authorization, promotion, completion notification, and the
spend cap remain binding checks.

Values cross from a tool call or another kernel by hand, because a callable
cannot. The check still binds. `preflight` compares the declared workspace
against the name the identity read reported and refuses when they differ, and
it compares every declared environment binding against the ledger. Bind
`details_reader` alone and the ledger measures but the workspace does not, so
authorization stays `not measured` and blocks.

If a Modal job returns `Modal authorization failed`, re-authenticate your Modal session in Settings under Compute to refresh the account token.

Source: [`provider_authorization.py`](../claude_binder/provider_authorization.py) measures authorization. [`modal_preflight.py`](../claude_binder/modal_preflight.py) defines the free ordered checks. [`modal_platform.py`](../claude_binder/adapters/modal_platform.py) executes job submission, notification settlement, and artifact promotion.

## fal authorization

fal stages read one named environment variable from the process that runs
preflight and execution. The default name is `FAL_KEY`. If Claude Science
injects the same credential under another name, set the selector before calling
Binder:

```python
import os
os.environ["CLAUDE_BINDER_FAL_CREDENTIAL_ENV"] = "MY_FAL_CREDENTIAL"
```

The selector contains a variable name. The credential value stays in the
host-provided variable and never enters argv or Binder's logs. The same selector
controls authorization preflight and every packaged fal client. A workstation
credential wrapper remains available when the process has no credential
variable, and that wrapper supplies its own `FAL_KEY`.

Every packaged client also takes `--credential-env <NAME>` directly, which is
the shorter route when driving a client as a subprocess rather than running a
campaign stage.

Check which name holds the credential before reading a 401 as a bad key.
Measured in a Claude Science session on 2026-09-04: the session exported a
`FAL_KEY` whose value was 74 characters, contained whitespace, and had no colon,
while the credential that actually authorized the application was stored under a
differently named secret. The first request returned

```
fal request failed with HTTP 401: {"detail":"Cannot access application
  \"<team>/<app>\". Authentication is required to access this application."}
```

and `--credential-env <that other name>` cleared it with no other change. A fal
key is two colon-separated fields, a 36-character id and a 32-character secret,
so the shape is checkable without printing the value:

```python
import os
v = os.environ.get(NAME, "")
print({"len": len(v), "has_colon": ":" in v,
       "parts": [len(p) for p in v.split(":")] if ":" in v else None,
       "whitespace": any(c.isspace() for c in v)})
```

A value that fails that shape check is not a fal credential, whatever it is
named. Read the 401 guidance below only after the shape check passes: refreshing
a secret that was never the fal key changes nothing.

The default fal authorization check sends no request. To measure a selected fal application, add
`--allow-fal-authorization-probe` to `capabilities`, `preflight`, and `execute`. The flag sends a
queue-status request for the selected application. It starts no campaign stage. One measured call
of the queue-host form answered HTTP 404 in 0.269 s on 2026-09-08, which is too fast to be a runner
cold start. No invoice was read for it, so its price is not established, and the
package keeps this explicit opt-in.

An HTTP 401 means that the credential visible to this process is invalid for the request. Refresh
the Claude Science secret or inject the same credential source that the working wrapper uses. An
HTTP 403 means that fal recognized the credential and refused access to the application. Confirm
the application account and access policy before asking its administrator to change permissions.

Use the same materialized plan for preflight and execution:

```python
assert binder_cli([
    "preflight",
    "--plan", str(bundle / "run-plan.json"),
    "--config", str(bundle / "config.resolved.json"),
    "--allow-fal-authorization-probe",
    "--json",
]) == 0

assert binder_cli([
    "execute",
    "--plan", str(bundle / "run-plan.json"),
    "--allow-fal-authorization-probe",
    "--json",
]) == 0
```

The probe flag does not replace the plan-bound approval or USD ceiling. Execution still checks
both before a paid stage starts.

The ESMFold2-Fast adapter applies a local wall limit to the complete client
process. Before each prediction it records a stable exact-call identity in a
hash-chained journal under the run artifact root. A timeout stops the local
client and leaves that call unknown across new attempts and changed local wall
limits. The synchronous endpoint supplies no provider request ID that Binder
can query or cancel after the client exits, so operator evidence must reconcile
that exact call before it can run again. Other call identities remain eligible.

## RunPod authorization

The shipped RunPod client can list the account's Serverless endpoints. Bind
`configure_binder_runpod_platform` with that read-only client or another
read-only account reader. An authorized result proves the selected credential
can read the account. It does not prove an endpoint ID, worker image, worker
contract, artifact harvest, or rate record.

An unbound reader or ambiguous response is `not measured` and blocks paid
Binder dispatch. A conclusive denial is `refused`. A direct RunPod workflow
uses the provider's own authorization and can hand artifacts back to Binder.
Read [RunPod route](runpod-route.md) before presenting the Binder route as
available.

## Lambda Cloud authorization

Lambda Cloud can use the same four authorization states after a caller supplies
a verified host and read-only account callback. The callback receives
`{"provider": "lambda", "mode": "read"}` and must affirm access without
allocating compute.

No such Lambda Cloud account callback or provider transport ships in this
package. The Lambda template therefore has no Binder authorization result to
present until the caller implements and verifies it. A direct Lambda workflow
uses the provider's own authorization and can hand artifacts back to Binder.
Read [Lambda Cloud route](lambda-route.md) before treating the template as a
dispatchable Binder route.
