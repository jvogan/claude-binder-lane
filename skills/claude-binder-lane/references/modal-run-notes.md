# Modal runs: field notes

The opening observations came from one Claude Science frame on 2026-08-31.
Later guarded runs supersede its package-status claims; one of them completed
all 16 stages. Numbers marked TODO were not measured.
Two values in the samples below are placeholders. Read `<your-image-id>` from your own `build_env()` result and `<your-workspace>` from a Modal `TokenInfoGet` call in the `compute_provider` kernel. Neither is shipped, because both name one account's resources.

---

## 0. Standalone inspection fold

This direct host call is an inspection example. It does not create a Binder
campaign receipt, enforce the campaign cap, or prove the guarded transaction.
Use the guarded dispatcher for campaign work. A scientist who deliberately
wants one standalone structure can use the host surface and record it outside
the campaign:

```python
# repl tool (host.compute lives there, not in the python kernel)
c = host.compute.create('modal', provider_params={
    'image': '<your-image-id>', 'env': 'esmfold2_gpu', 'gpu': 'A100-80GB',
    'cpu': 8, 'memory': 32768,
    'volumes': {'/datavol_esm': '<volume-from-live-environment-ledger>'}, 'timeout': 3600})
job = c.submit_job(command='python fold.py', inputs=['fold.py', 'spec.json'],
                   outputs=['fold-receipt.json', 'structures/*.cif'],
                   intent='ESMFold2-Fast co-fold', run_timeout_s=1500)
# end the cell, end your turn, then wait_for_notification
```

The `esmfold2_gpu` image already has ESMFold2, ESMFold2-Fast and ESMC-6B weights on the Volume with
`HF_HUB_OFFLINE=1`. Nothing downloads at job time. Three 186-193 residue two-chain folds took 53 s
of job wall including a 21.3 s model load.

In the container, follow Modal's published example shape:

```python
from esm.models.esmfold2 import ESMFold2InputBuilder, ProteinInput, StructurePredictionInput
from transformers.models.esmfold2.modeling_esmfold2 import ESMFold2Model
model = ESMFold2Model.from_pretrained("biohub/ESMFold2-Fast", revision=REV).cuda().eval()
res = ESMFold2InputBuilder().fold(model, StructurePredictionInput(sequences=[
    ProteinInput(id="A", sequence=target), ProteinInput(id="B", sequence=binder)]),
    num_loops=3, num_sampling_steps=50, num_diffusion_samples=1, seed=0)
cif_text = res.complex.to_mmcif()          # there is no .to_pdb()
plddt, ptm, iptm = float(res.plddt.mean()), float(res.ptm), float(res.iptm)
```

Three traps in that snippet:

- **`res.complex.to_mmcif()`.** The result object has no `to_pdb`. Serialize from `.complex`.
- **Confidence values are fractions 0-1**, not 0-100. `pLDDT 0.90` is good, not terrible.
- **`num_loops=3, num_sampling_steps=50` is the documented example's budget, not the library's
  default.** `fold()` defaults to `num_loops=20, num_sampling_steps=200`. Modal's example passes the
  smaller values. Pass them explicitly so your receipt records what you actually ran. Whether the
  larger budget changes these designs' scores is TODO.

You cannot preflight the model on CPU. `import transformers.models.esmfold2` autotunes fused
kernels at import and raises `RuntimeError: 0 active drivers` with no GPU. A CPU inventory job can
read `esm`, `torch`, `transformers` versions and walk the HF cache, but it cannot touch the model
class. Budget one cheap CPU job for inventory if you want it; don't expect it to validate the fold.

---

## 1. Kernel map

| you need | `python` kernel | `repl` (control-plane) |
|---|---|---|
| `host.compute` | **no** (`capabilities()['compute'] is False`) | **yes** |
| `host.artifacts`, `host.lineage`, `host.llm` | yes | yes |
| import `claude_binder` / `dispatch_modal` | yes (3.11) | yes, **Python 3.9.6**, with one fix |
| read a granted host path (a directory outside the workspace) | yes | **no**, `PermissionError` |
| pandas / numpy | yes | no (stdlib only) |

Consequences, all measured:

1. **The dispatcher must run in `repl`**, because that is the only kernel with `host.compute`.
2. **`repl` is Python 3.9.6.** Loading a package there by file path fails with
   `AttributeError: 'NoneType' object has no attribute '__dict__'` from `dataclasses`. Register the
   module *before* executing it:

   ```python
   spec = importlib.util.spec_from_file_location("dispatch_modal", B + "/scripts/dispatch_modal.py")
   dm = importlib.util.module_from_spec(spec)
   sys.modules["dispatch_modal"] = dm        # <-- required on 3.9, before exec_module
   spec.loader.exec_module(dm)
   ```

   With that one line the whole `claude_binder` package imports and runs under 3.9.
3. **`repl` cannot read a granted host path.** `build_bootstrap_spec` needs
   `<source_repo>/src/claude_binder`, so use a `bash` cell to copy the installed skill's
   `claude_binder/` directory into that layout inside the frame workspace. Verify the complete file
   manifest against the installed package before dispatch; do not rely on a remembered file count
   or a single-module hash.
4. `lane execute` in a subprocess has **none** of the kernel injections. See §4 item 2.

**Path masking.** `pwd`, `os.path.realpath` and any printed path render a workspace subdirectory as
`./binder/...` while the string on disk is genuinely absolute (142 and 178 chars, both starting
`/`). Check `len(p)` and `p.startswith('/')`; never "fix" a path that only looks relative.

---

## 2. Campaign happy path, in order

Provider work starts after the free preparation steps. Follow the steps that apply to the selected
route in order; the offline contract check in step 6b is conditional on compatible prior artifacts.

1. **`compose`**, `lane compose --campaign campaign.json --profile profile.json --out composed.json`
2. **`materialize` pass 1**, into `<run_root>/run-bundle`, purely to give the spend gate a plan to
   price. Produces `freeze_digest: null`, which is why approval will refuse.
3. **`run_intent.py`** with `--say <plain sentence> --plan ... --tier modal --rate-record ...
   --ledger <bundle>/approvals.jsonl`. Prints the freeze digest, resolves paid stages, derives the
   effective cap as `min(campaign cap, account policy cap)`. **Refuses**: "the plan has no valid
   freeze digest."
4. **`materialize` pass 2** with `--freeze-digest <digest from step 3>`. Move the pass-1 bundle
   aside first, because `materialize` refuses with `output directory is not empty` and its
   `replace` flag is reserved and does nothing:
   `mv <run_root>/run-bundle <run_root>/run-bundle.pass1`. Keep it rather than deleting it. It is
   the evidence for where the freeze digest came from.
5. **`run_intent.py` again** on the new plan → `APPROVED AND WRITTEN`.
6. **Free stages, one `--stage` at a time**: `target-prepare`, `runtime-check`,
   `normalize-candidates`, `filter-integrity`, `filter-novelty`. Start with no `--resume`; use
   `--resume` for later calls against that run root. Five stages, not six. `stage-msa` is free but
   on the supplied-candidate profile it is stage 7, after the paid screen.

   The plan is a dependency graph. A dependency-ready stage may checkpoint even when it is not the
   next displayed stage. A request with an unmet predecessor refuses before it runs. Read
   `stage-checkpoint.json` for the committed state. On resume, the executor reconciles valid stage
   receipts into that checkpoint rather than discarding work that a prior invocation completed.
6b. **Optional free contract check when compatible artifacts exist.** It runs the whole graph to
   `render-viewer` offline against a run plan and a completed artifact tree, calls the
   real adapters, and withholds only the paid predictor call behind a stub that fails closed.
   It needs paid-stage artifacts and the first it reads is `artifacts/scores/screen-score-table.jsonl`,
   so use it only when a compatible receipt-backed completed run is available. A first target or
   first route has no such source; in that case the guarded one-candidate smoke barrier is the first
   executable proof and this check is not a prerequisite. After the screen completes, use this run's
   `artifacts/` tree ahead of `promote`:

       PYTHONPATH="$BINDER_SKILL" python3 -m claude_binder.contract_dry_run \
         --plan <new_run_root>/run-bundle/run-plan.json \
         --artifact-root <completed_run_root>/artifacts \
         --from normalize-candidates \
         --fixture-root <scratch>/dryrun

   Expect `ok: true`, `findings: []`, `provider_calls: 0`, fourteen stages. It takes under a minute
   and costs nothing. `--artifact-root` and `--fixture-root` must be separate. When this optional
   check is run, resolve its findings before dispatch. This coverage applies to the supplied-candidate downstream path. It does not
   invent generic runners for alternate predictor, generator, or optimization routes: a selected
   stage without coverage reports a failure. Adapter progress goes to stderr so stdout stays
   parseable.

7. **Dispatcher, in `repl`**: `verify` → `plan_stage` → `build_bootstrap_spec(attempt_id=...)` →
   `submit_wave` → end turn → `wait_for_notification` → `collect_notifications` → `close_wave` →
   repeat for the scientific wave with the **same** `attempt_id`.

8. **Terminal pass after every stage is committed.** Run an unfiltered `execute --resume` against the
   same plan and run root. With no pending stages, this is local terminal validation: it validates
   the bundle, receipts, artifacts, and terminal report, but does not run a Modal preflight or
   submit a job. If a stage remains pending, use the `repl` dispatcher context for the selected
   Modal work instead.

`materialize` must write the bundle **inside the run root** (`--out <run_root>/run-bundle`). Its only
warning about the alternative mentions durability, but the real consequence is that
`candidate_normalizer` fails with `source manifest override is missing`.

### Readers you must bind

`submit_wave` re-measures authorization and needs two callables. Neither exists as a kernel method:

```python
def details_reader(request):   # request is {'provider':'modal','mode':'read'}
    return {"details": LEDGER_TEXT}      # verbatim ### env: blocks from the compute_details tool
def identity_reader():
    return {"workspace_name": "<your-workspace>"} # from a live Modal TokenInfoGet read
```

The parser only consumes `### env:<name>@<spec_sha>` headers and `image_ref:` fields, so relaying
those blocks verbatim is faithful. `compute_details` never reports a workspace name, that is why
`identity_reader` exists. Get it live in the `compute_provider` kernel:

```python
_Client._client_from_env = None; _Client._client_from_env_lock = None   # reset the cached singleton
r = await (await _Client.from_env()).stub.TokenInfoGet(api_pb2.TokenInfoGetRequest())
```

Without that reset, `from_env()` returns a cached singleton whose `_stub` is already set and the read
wedges. This is not a DNS or network problem, and looking for one wastes a frame.

---

## 3. What "working" looks like at each checkpoint

Stop and read the error if you don't see these exact shapes.

| step | expected |
|---|---|
| `verify` | `ok: true`, `errors: []`, `modal_stage_ids` listed, `adapters_without_measured_seconds_per_candidate: []` |
| authorization probe | `status: authorized`, one binding `<env> <spec_sha> <image> -> authorized` |
| `plan_stage` | an `attempt_id`, `mode` (`single` or `smoke_scale`), `waves: [1]` or wider |
| bootstrap job | `exit_code: 0`, `bootstrap-result.json = {"ok": true, "file_count": <N>}` where `<N>` matches the staged full-manifest count |
| `collect_notifications` | **returns 1 row per job.** Zero rows means you passed the wrong shape, §4 item 4 |
| `close_wave` | `barrier: true`, `volume_barrier: true`, `still_open: []` |
| GPU wave submit | returns in seconds. 6.4 s measured. A submit that hangs ~1800 s is the old defect |

`stages_whose_timeout_exceeds_the_container` in `verify` output is informational, the stage's plan
ceiling exceeds `container_timeout_s` and the dispatcher applies the runtime ceiling.

---

## 4. Gotchas: symptom → cause → fix

1. **`already registered; attach or resume it instead of resubmitting`, then `authorize-retry` says
   `prior paid attempt ... is not recorded`.** `submit_wave` writes the phase claim to the job
   register *before* the provider accepts, so a failure during **local** staging leaves a row with
   `job_id: null` that blocks resubmission and is invisible to the retry gate. Every documented
   recovery (attach, resume, `bind-submission`, `authorize-retry`) assumes a provider job exists.
   **Fix:** start a new run (`run_id`), which is a clean re-materialize, not a gate bypass.
   Nothing was billed. **Do not** hand-edit the register to get past it.

2. **`lane execute` reports an unbound Modal capability.** The selected paid
   stages need a live host, environment resolver, completion notification, and
   artifact promotion. A selected image stage also needs `render_smoke`; a
   binding that declares packages needs `package_resolver`. These come from
   `platform_context`, which a plain CLI does not populate. Bind the capabilities
   in the Claude Science kernel and rerun preflight. Unselected catalogue tools,
   generator bindings removed by a supplied-candidate profile, and an unused
   renderer do not block the run.

3. **A plan carrying the `build_env()` `spec_sha` is refused.** `probe_modal_authorization` compares
   the plan against the `compute_details` ledger (`entry["spec_sha"] != binding["spec_sha"]` →
   REFUSED), and the two sources report different values. Measured:
   `build_env('esmfold2_gpu')` → `b84e2babc270ab7f`, while the `compute_details` ledger header →
   `7305ca31ec661bc0`, **same image id in both**. Write the ledger value into
   `environment_identity: modal-env:<env>@spec_sha=<ledger sha>`. TODO(evidence): which of the two
   hashes describes the built image. Settled by: a rebuild that reports both values.

4. **`collect_notifications` returns 0 rows and `close_wave` says `awaiting_completion` forever.** It
   gates on `notification["notification_type"] == "compute_done"` and its docstring says "payloads".
   Pass the **whole envelope**, not `notification["payload"]`:
   `dm.collect_notifications(RR, [{"notification_type": "compute_done", "payload": payload}])`.

5. **`ENOENT` on `bootstrap.sh` at submit.** `submit_wave(workspace=...)` wants the **frame workspace
   root**; the dispatcher appends `policy.workspace_staging` ("dispatch") itself. Passing the staging
   directory stages to `dispatch/dispatch/...`. This then triggers gotcha 1, so get it right first
   time.

6. **`dispatch policy has unknown fields`.** The policy schema is strict. Provenance for a measured
   number (e.g. where `seconds_per_candidate` came from) must live in a sidecar file, not in
   `dispatch-policy.json`.

7. **`volume_mount` must be absolute** for Modal, and it should equal the `--data-root` so in-container
   paths resolve. It will *print* as relative, see §1 path masking.

8. **Give `build_bootstrap_spec` the installed package in its expected layout.** It requires
   `<source_repo>/src/claude_binder`, while an installed skill keeps the package at
   `<skill>/claude_binder`. In a `bash` cell, create a workspace-local synthetic source root and
   copy the installed package there. Compare the complete staged manifest against the installed
   package before dispatch, on relative paths, byte counts and SHA-256 values.

---

## 5. Cost hygiene

Billing tracks sandbox lifetime, not job wall time. Measured: **63 s of A100 job wall against
0.70721154 USD of A100 billing** in the same window. A handle's sandbox stays warm and
self-terminates only after ~15 min idle, so several short jobs on separate handles bill far more than
their runtime.

- `c.close(intent=...)` on **every** handle when its last job is done. `close_wave` closes only the
  handles in that wave.
- `host.compute.ledger()` before you finish, expect `sandboxes (0)` and `workdirs (0)`.
- The `WorkspaceBillingReport` buckets by **app and hour**, and the app is shared across sessions in
  a workspace. Per-job attribution is not recoverable from it. Report app totals, not per-job costs.
- The report lags by hours. A finished job with no settled figure is normal; do not re-run GPU work
  to chase a provider number.
- **Modal's `compute_done` payload carries no usage**, so `provider_usage_reported` is always `False`
  and `charges: []`. `close_wave` handles this correctly now: it writes a settlement-unavailable
  close-out record with
  `amount: 0.0`, `amount_source: "Modal attempt closed with no per-job usage reported"`,
  and `settlement_unavailable: true`. This is not a settled charge or a reconciliation row. The
  reservation remains outstanding against the cap. A campaign will therefore show
  `cumulative_settled: 0.00` with a non-zero estimate. That is expected.

---

## 6. Values that are resolved a second time

Two values that look settled in a rendered command or a materialized plan are derived again at run
time.

**`target_sequence` is resolved again inside the container.** `control_builder` builds the control
fold's predictor command through `predictor_command_context`, which calls
`lane.resolved_context(config, require_residue_map=True)` on the config the adapter was handed. Its
resolver, `_target_sequence_value`, prefers `runtime_structure_path`, the bundle copy, and picks the
first structure field whose file exists. `structure_source_path` names the operator's own file outside the run bundle, so
that path is absent on the Volume, and a resolver that read the first named field instead would
return no sequence and fail `control-calibration` with
`control builder: ValueError: command token has no value: target_sequence`. The shard rendered on
the host carries the full value, for example `--target-sequence pdl1-a-18-132=AFTVTVPKD...`, the
full 115 aa, because every structure path exists on the host. A populated flag on the shard does
not mean the value survives inside the container. Grep an adapter for `lane.resolved_context`
before treating a rendered command line as settled.

**A fan-out count can come from a stage that ran locally.** `cofold-screen-esmfold2-fast` carries
only `fanout.count_from: {stage_id: filter-novelty, artifact_id: passing-candidates}`, and no
campaign or profile field pins `fanout.scale_count` for that stage. `resolve_scale_count` reaches
`find_stage_receipt`, which walks the job register for a Modal job (phase `finalize`/`single`,
`state == succeeded`) and reads `hpc/<job_id>/receipt.json`. `filter-novelty` runs locally, so it
has no register row and no harvested receipt, and the resolver falls back to the local run-root
receipt. `verify` reports a plan like this as mixed-route and lists `filter-novelty` in
`ignored_non_modal_stage_ids`, which is informational. Do not edit a materialized plan to pin the
count.

---

## 7. Measured reference numbers

Use these for planning; replace them when you measure your own.

| quantity | measured |
|---|---|
| ESMFold2-Fast model load, A100-80GB | 21.3 s |
| ESMFold2-Fast fold, 186-193 res two-chain | 1.6 - 3.4 s |
| `seconds_per_candidate` bound used | 10 s (rounded up from the above) |
| bootstrap job | exit 0, 1 s, 222 files host→Volume |
| GPU wave submit latency | 6.4 s |
| free stages (5, local) | seconds each |
| app billing, 6 h window | 1.30726245 USD total, 0.70721154 USD A100 |
| campaign cap / estimate / settled | 5.00 / 0.6639035619634 / 0.00 USD |

One guarded run dispatched six Modal jobs, all exit 0, and folded five design complexes and both
controls.

GPU seen: `NVIDIA A100-SXM4-80GB`, 85094825984 bytes. Container: Python 3.12.1, torch 2.7.1+cu126,
transformers 4.57.6, esm 3.3.0, numpy 2.4.6, `Linux-4.19.0-gvisor`.
Weights: `biohub/ESMFold2-Fast@b28d8ace5e05e61e5bec1e6820cfd3e221819d12`,
`biohub/ESMFold2@1ebf0e3481a5184eb6171d40615c79e384b48796`.

Anything a run produces is a computational prediction. Structure and interface confidence numbers
establish nothing about binding, affinity, function or selectivity.
