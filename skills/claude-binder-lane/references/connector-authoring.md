# Connect a selected tool

Claude Science can write a campaign-local connector for the scientist's chosen
tool. Reuse an existing adapter when its request and output contract matches
the selected service. Otherwise, adapt that contract in the campaign workspace.
The tool catalogue is a starting set for discovery.

## Inspect the contract first

After loading the Binder skill, inspect a packaged profile in the Python kernel:

```python
from claude_binder.connector_contract import inspect_contract
from claude_binder.paths import package_file

profile = package_file("data", "templates", "profiles", "full-ensemble.template.json")
menu = inspect_contract(profile)
[(item["adapter_id"], item["modules"]) for item in menu["contracts"]]
```

Select an adapter ID returned by that list:

```python
contract = inspect_contract(profile, "solublempnn-designer")["contracts"][0]
contract
```

The result includes the inherited adapter, attached stages, all three command
templates, output schemas, support files, and unresolved adapter and stage
fields. Inspection performs no toolcheck, network call, or dispatch. Campaign
inputs, route compatibility, and authorization remain separate checks.

For a shell, set `BINDER_SKILL` as described in the [free first run](free-run.md):

```sh
PYTHONPATH="$BINDER_SKILL" python3 -B -m claude_binder.connector_contract \
  --profile full-ensemble.template.json --adapter solublempnn-designer
```

`--profile` also accepts a local profile or composed configuration. Use the
actual selected profile: a provider overlay can replace the command and parser
behind the same adapter ID.

## Decide what needs changing

| What you have | Next action |
| --- | --- |
| Matching adapter, accessible image or endpoint, and matching input/output contract | Resolve remaining declarations from the live environment, then qualify the selected configuration. Reuse cached weights. |
| Accessible service with a different API | Write a request/response bridge. Keep its service revision and scientific model revision separate. |
| Matching adapter with no accessible deployment | Use the selected provider's environment setup, hydrate weights, and qualify that deployment. |
| A tool without a matching adapter | Copy the nearest role's contract and implement its run and parse commands. Use the selected tool's own input/output specification. |

For an entirely new tool ID, put a sourced row in the campaign JSON under
`tool_catalog_extensions`. Claude Science can do this in the scientist's
workspace without modifying the installed skill. The row is carried into the
composed configuration, run identity, and licence evaluation. It cannot replace
one of the packaged tool IDs. A bridge for an existing tool ID can use
campaign-local support files without a new row.

```json
{
  "tool_catalog_extensions": {
    "your-tool-id": {
      "display_name": "Name reported by its publisher",
      "stage_category": "cofolding",
      "code_licence": {
        "status": "unverified",
        "commercial_use": "unknown",
        "evidence": ["Replace with the official code licence citation"]
      },
      "weights": {
        "status": "unverified",
        "commercial_use": "unknown",
        "evidence": ["Replace with the official model weight terms citation"]
      },
      "gate": {
        "requires_written_agreement_for_commercial": false,
        "conditions": []
      }
    }
  }
}
```

Replace the example values with the selected tool's actual code and weight
terms before use. For a tool with no weights, set `weights.status` to
`not_applicable` and `weights.commercial_use` to `N/A`, citing the absence of
a weight layer. Unknown commercial terms refuse a commercial campaign until
resolved. The row establishes tool identity and terms; the adapter below
establishes execution. Provider and spend approvals apply to the new route as
they do to packaged routes.

A registered service proves that the host knows a service ID. It does not
prove compatibility with a package adapter. The shipped Boltz adapter invokes
`boltz-api`; a `boltz2-service` NIM endpoint uses a separate request contract.
The ProteinMPNN adapter uses a checkout runner or its fal client. A hosted
`proteinmpnn-service` response needs a bridge to its candidate manifest.
Use [NIM handoffs](bionemo-nim-route.md) for those native service surfaces.

## Write the campaign-local bridge

1. Preserve the selected stage's accepted artifacts, output schemas, candidate
   identifiers, chain mapping, and parser-result path. Compare toolcheck, run,
   and parse arguments together. Copy successful and failed provider responses
   into local replay fixtures with credentials removed.
2. Implement the smallest conversion: stage inputs to the native request,
   followed by native outputs to attempt-owned artifacts. Account for every
   requested candidate with a scored row or a recorded failure and reason.
   Reconcile missing, duplicate, and unexpected IDs before accepting a batch.
3. Put connector scripts beside the campaign configuration and declare them in
   `adapter.support_files`. Use those exact path strings as argv elements in
   `toolcheck_argv`, `command_argv_template`, and `parser_argv_template`.
   Materialization copies and hashes the files and rewrites those argv elements.
   Keep support-file basenames unique within an adapter.
4. Use the provider's native route while a Binder bridge is still being built.
   An automated Binder bridge needs a real transport, worker contract, job
   identity, output retrieval, and cleanup. Preserve admission-control
   estimates and resume checks. Select credentials by environment-variable name;
   keep values out of argv, configuration, and receipts. An HTTP response alone
   does not complete a stage.
5. Compose and materialize the campaign, then run `contract-audit` and
   `preflight` using the commands in the [free first run](free-run.md).
   Replay both response fixtures through the real parser before a paid canary.
   For remote jobs, check the input manifest's referenced files and retrieve
   declared output files into the host artifact store.
6. Within the scientist's approved route and ceiling, run one prediction and
   parse the returned coordinates and score inputs. Record the model revision,
   sampling semantics, MSA condition, artifact hashes, and cost evidence. Use
   target controls to qualify scientific claims before widening the cohort.

An exploratory native-tool run can use a documented artifact handoff before a
connector exists. Label that execution route in the result. An automated Binder
stage additionally needs the declared dispatcher and receipt contract.

## Reproduce, substitute, or explore

For an exact reproduction, track every requirement in the
[published comparison](published-campaign-comparison.md), including generator
coverage, designer checkpoints, named co-fold arms, filters, and selection rules.
Complete each missing binding with the procedure in this guide. Binding every
module is necessary but does not establish exact reproduction: verify the actual
source and weight pins, seed delivery, runtime, ensemble construction, controls,
filters and scale against the published protocol. Stochastic outputs may still
differ, so preserve both the intended settings and the observed results.

For a substitution, declare the replacement and keep its results useful.
ESMFold2-Fast, Boltz, and AlphaFold-Multimer can define a three-arm ensemble;
they do not match the named Fast, Full, and Protenix baseline. The
`published-three-mode-zscore` ranking-mode string selects a formula from the
enabled predictor modes and the lineages that supply them, and it requires at
least two lineages; it does not attest model identity or baseline fidelity. Three
modes of one lineage rank by `candidate-single-lineage-three-mode-raw-mean` under
a candidate claim, and are refused under a published claim. Record the actual arm
IDs, checkpoints, sampling semantics, and normalization cohort with that score.

A smaller exploratory campaign may use the single-method exceptions declared
by its profile. Expand method diversity when that serves the selected claim or
objective. A catalogue entry without local qualification remains available for
discovery, integration, and the scientist's own bounded validation.
