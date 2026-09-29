# Bringing up Modal compute

Four values must come from Claude Science before starting a Modal run. `build_env()` runs in the `compute_provider` kernel. `compute_details` is a Claude Science tool call whose returned value can be bound into the kernel; it is not a Python name in that kernel.

## 1. Read the connection

```text
compute_details({'provider': 'modal', 'mode': 'read'})
```

The result supplies the environment ledger, including each environment's identity and image reference. It does not supply the workspace name or current GPU price. Read the workspace through the separate identity call in [Provider authorization](provider-authorization.md). Record a price only from a provider price display or rate record that actually reports it.

For admission control, record that current price as an operator-attested planning rate with the exact hardware SKU, currency, source, and read time. A current provider price display can supply this record. It is not an authenticated receipt or settled bill, and the present rate loader does not enforce freshness. Keep later provider billing reconciliation separate from this pre-dispatch estimate.

Reuse an existing image and weight volume when their recorded specification,
model identity, and provider environment match the selected run. Read the
ledger before scheduling a build or hydrate. Reuse avoids repeating those
steps; it does not establish that storage or the next job is free.

## Recover the local provider helper

A missing `compute-provider-modal` interpreter or a failed `import modal` in
the provider kernel identifies a local integration failure. A working Modal
CLI in another environment does not verify Claude Science's helper. Inspect
the helper selected by the running platform release and its provisioning error.

The platform's bundled `remote-compute-modal/provider.json` declares the helper
environment and package pins; its `requirements.lock` records dependencies.
Read those installed files before claiming that the SDK version is unknown.
Use the platform's supported reprovisioning action when available. Preserve the
provider connection and environment ledger. Do not replace the helper with an
arbitrary SDK install or assume that a moved-aside environment is compatible.

If recovery requires an app restart or a Settings change the session cannot
perform, explain that exact action and any interruption it causes. Verify the
helper and provider connection after recovery. A restart suggestion alone is
not evidence of repair. Continue independent work already covered by the plan
and retain completed artifacts for resumption.

## Current paid-execution boundary

Real direct Modal execution through `lane execute` is refused. That synchronous branch can render a stage, and it does not provide the complete remote smoke barrier, bounded scale waves, parser/finalizer, canonical receipt, and atomic spend/resume contract. Dry-run, validation, materialization, authorization measurement, and preflight remain available.

Two kernel facts make that refusal structural rather than a policy the caller can opt out of, measured in a live session on 2026-08-31. `binder_cli` runs in the analysis kernel, and there `host.capabilities()` reports `compute: false`, so `modal_platform.submit_stage` finds no `host.compute.create` to call. `save_artifacts` is not a name there either. Completion is the harder half: `modal_platform._matching_notification` calls `wait_for_notification()` in a loop, synchronously, inside the executing process, and Claude Science delivers that notification to the agent between cells rather than to a blocking Python callable. So a Modal stage started from `execute` has no way to learn it finished. Use the guarded wave dispatcher.

The separate `dispatch_modal.py` path applies the runtime ceiling to explicit as well as receipt-derived fan-out and caps shard width, concurrency, and job timeouts. Its CLI never submits. The kernel-only `submit_wave` re-measures Modal authorization with the live `compute_details` reader, serializes the final cap read, reserves the full approved stage maximum before creating a handle, and binds every submission to one stage attempt. Bootstrap, smoke, scale, and finalize share that attempt ID and reservation. A second kernel cannot resubmit an already claimed phase/shard.

Plan the first Modal stage before building its bootstrap specification. Pass the plan's `attempt_id` to `build_bootstrap_spec`; submit, collect, and close that bootstrap wave before submitting the first scientific wave. Reusing the attempt ID is required: it keeps the CPU bootstrap and the stage under one approval and prevents a second reservation. The closed bootstrap is an allowed opening barrier for either a `single` or `smoke_scale` stage. It is not a free or out-of-band setup call.

`close_wave` reads the terminal provider result before closing, records settled usage, closes every handle to commit the Volume, and clears the estimate only after every job in a terminal attempt has a provider charge. Its returned barrier is the conjunction of Volume and financial barriers. A retry uses a new attempt ID and needs an append-only, hash-chained authorization written by the CLI after the prior attempt is terminal, closed, and financially reconciled:

Both shell blocks below read three paths. `BINDER_SKILL` is the installed skill directory,
the one that holds `kernel.py`. `BUNDLE` is the materialized run bundle. `RUN_ROOT` is the
run directory the attempt writes into.

```sh
export BINDER_SKILL="/path/to/installed/claude-binder-lane"
export BUNDLE="/path/to/run-bundle"
export RUN_ROOT="/path/to/run-root"
```

```sh
PYTHONPATH="$BINDER_SKILL" python3 "$BINDER_SKILL/scripts/dispatch_modal.py" authorize-retry \
  --plan "$BUNDLE/run-plan.json" \
  --run-root "$RUN_ROOT" \
  --stage '<stage-id>' \
  --prior-attempt '<prior-attempt-id>' \
  --new-attempt '<new-attempt-id>' \
  --authorized-by '<operator identity>' \
  --reason '<reviewed reason>'
```

Two kinds of evidence back these transaction rules: automated tests that exercise them against simulated provider responses, and a guarded Modal campaign that committed all 16 stages. That live run exercised remote submission, artifact return, local continuation, ranking, and viewer output. Modal supplied no per-job settled amount, so billing settlement remains unproven and must not be represented as zero cost.

If `submit_job` raises after the provider may have accepted work, the dispatcher records a durable `submission-uncertain` intent and blocks later waves. Inspect the Modal ledger. When it shows the accepted job, bind that exact ID and use the normal reattach path; do not resubmit the shard:

```sh
PYTHONPATH="$BINDER_SKILL" python3 "$BINDER_SKILL/scripts/dispatch_modal.py" bind-submission \
  --plan "$BUNDLE/run-plan.json" \
  --run-root "$RUN_ROOT" \
  --submission-id '<submission-id from jobs.jsonl>' \
  --job-id '<job-id from the provider ledger>'
```

The `spec_sha` and `im-` image ID are scoped to one Modal environment. Read them on the workspace where the campaign will execute.

## Qualify the arm before the first stage

`runtime-check` is one graph stage. It refuses an enabled production-scoring Modal arm whose roster row is not `PASS` for this campaign's own target. The guarded dispatcher separately rechecks authorization, approval, and spend before each paid attempt. Qualify a production arm before that graph stage. An exploratory profile can run while preserving its unqualified claim status. [Qualify a Modal arm](modal-qualification.md) gives the receipt contract and the frame procedure.

## 2. Build the environment

```python
r = build_env('esmfold2_gpu', hydrate=True)
print(r)
```

Setting `hydrate=True` downloads environment weights idempotently.

The result returns `image`, `spec_sha`, `volumes`, `env`, `gpu_default`, `egress_domains`, `hydrate_defined`, and `hydrated`. Its `spec_sha` is not the value provider authorization compares, and it differs from the ledger's. Authorization reads the environment ledger, so after building read `compute_details` again and populate both fields from it:

| From the `compute_details` ledger | Profile field | Form |
|---|---|---|
| environment header `spec_sha` | `environment_identity` | `^modal-env:[^@\s]+@spec_sha=[0-9a-f]{16,64}$` [runtime_validator.py](../claude_binder/adapters/runtime_validator.py) [generator_preflight.py](../claude_binder/generator_preflight.py) |
| the same environment's `image_ref` | `resources.container_image_digest` | `^im-[0-9A-Za-z]+$` [modal_platform.py](../claude_binder/adapters/modal_platform.py) |

The field name is a trap. A container registry SHA-256 digest is the wrong value for `resources.container_image_digest`.

Record `egress_domains` in session notes to reuse the cached image without re-running `build_env()`.

`scripts/bootstrap_environments.py` does this for a whole profile instead of one
environment at a time. Pass it a profile and it prints the plan, which is its
default and makes no build call:

```sh
python3 -B scripts/bootstrap_environments.py \
  claude_binder/data/templates/profiles/modal-platform.template.json
```

That profile plans `proteomics_gpu`, `esmfold2_gpu` and
`proteomics_rfd_diffdock_gpu`. Add `--build` and call `run_bootstrap` from a
`repl` cell with the pre-bound `host` to execute them, because the command line
never assumes the host object is importable. Results append to a JSONL ledger,
and resolved values go to a separate JSON report rather than back into the
profile.

The returned `volumes` are required at submit time. The `esmfold2_gpu` image runs with `HF_HUB_OFFLINE=1` and holds no weights itself; they live on the Volume. A `submit_job` whose `provider_params` names the image, env and GPU but omits `volumes` starts a container that fails at the first `from_pretrained`, measured 2026-09-04:

```
huggingface_hub.errors.LocalEntryNotFoundError: Cannot find the requested files
  in the disk cache and outgoing traffic has been disabled.
OSError: We couldn't connect to 'https://huggingface.co' to load the files, and
  couldn't find them in the cached files.
```

The job exits 1 in about nine seconds, harvests zero files, and still starts a billable sandbox. Carry the Volume mapping into every submission:

```python
provider_params={..., 'volumes': {'/datavol_esm': 'claude-science-esm-cache'}}
```

[Modal run notes](modal-run-notes.md) §0 shows the full parameter block. A recipe that records image, env and GPU but drops the Volume costs one failed submission to rediscover.

## 3. Environment coverage limits

One environment does not run the full lane. The `esmfold2_gpu` environment pins Python 3.12, whereas other bundled proteomics environments pin Python 3.11. A Modal run uses separate containers per adapter group.

The bundled `binder_scoring_gpu` recipe records a pinned Protenix v2 source and official hydration route, but deliberately refuses to build until the official checkpoint receipt, dependency lock, completed image recipe, and offline GPU qualification exist. Record the arm as unrun until those facts are supplied. The deployed ESMFold2-Full and ESMFold2-Fast modes form one lineage.

## 4. Size the first job before widening

The first job runs one candidate with a lowered stage ceiling. Until an adapter has a measured `seconds_per_candidate`, `dispatch_modal.run_timeout_s` returns the full stage ceiling.

Read `res.wall_s` from that single-candidate run, record it in `dispatch-policy.json`, and then dispatch wider batches.

## Unresolved values before bring-up

`compose` accepts `__REQUIRED__` placeholders, while `materialize` refuses placeholders and reports the offending JSON path [lane.py](../claude_binder/lane.py). A run refuses execution on unpopulated image IDs, prices, or environment hashes.

Composing the demo campaign against `modal-first-run-floor` leaves 50 unresolved values. Forty are the identity and image pair across 20 adapters. Those 20 adapters name three environments: `proteomics_gpu` on 17, `esmfold2_gpu` on two, and `rfdiffusion_generator_gpu` on one. So 40 slots carry 6 facts you read once.

## What the modal_platform block sets

Two fields refuse a run, two select or display, and three are records the executor never reads. The
`modal_platform` block in a profile template carries all seven. `modal-platform.template.json` is the
worked example.

| Field | What preflight does with it |
| --- | --- |
| `bindings` | Refuses. Every selected Modal stage needs one binding, matched on `adapter_id` and, when an adapter serves several stages, on `stage_id`. Two exact bindings for one stage stop preflight. |
| `promotion_destination` | Refuses. An unset value fails the `persistence` check, because the workspace is deleted six hours after your last action. |
| `picture_owner_stage` | Selects. Names the stage to probe when more than one selected stage writes an image. |
| `approval_briefing` | Displays. The lines a session shows you before the two approval cards. |
| `roster_tools` | Warns. An entry that does not resolve records a warning and passes. |
| `workflow_bindings` | Records which decisions your session handles and which run on Modal. |
| `environment_catalog` | Records the environments, GPUs, and egress domains the bindings draw from. |

Each `roster_tools` entry names a **Claude Science skill**, resolved against the skill names installed
in your session. It does not name a catalogue tool, and it selects no adapter. Write an entry only for
a decision a platform skill can serve. The recorded platform snapshot ships no backbone generator and
no structure viewer, so `generate` and `pictures` have no skill to name. Pictures come from this
package's own `structure-picture-renderer` adapter on the `render-structure-pictures` stage.

`workflow_bindings` and `environment_catalog` document the plan for a reader. Changing either one
changes no execution. Change `bindings` to change what runs.

## Modal platform behaviours

- A Modal tag value is 63 characters or fewer. It accepts only letters, digits, dashes, periods, and underscores. Keep a derived run ID inside that value.
- A Modal billing report lags container exit by one to two minutes. A run whose outputs, hashes, and cleanup pass is a success with a pending cost when the report has not arrived. Do not rerun GPU work to chase a provider number.
- Ranking a whole round after one targeted stage fails because the surrounding stages did not run. Report a targeted stage on its own.
