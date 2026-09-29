# Cost calculation details

Calculate a campaign from its resolved graph, selected provider rate, bounded
runtime, and job receipts. This page gives the fanout calculation and recorded
Binder observations.

The packaged Modal example has 10 selected candidates and 84 co-folds, counted
by the shipped estimator. These counts describe that profile's selected
generator, controls, predictors, and seeds.

`RECORDED_COST_KINDS` in `claude_binder.qualify` distinguishes a settled billed
amount, provider-reported rate, modeled estimate, and unpriced execution. A
modeled total is not a provider bill. Claude Science's frame `total_cost` is
model-price telemetry; it does not establish a provider-compute charge.

## Plan from the resolved graph

Use `estimate_fanout` for the selected profile's workload and
`estimate_from_graph` for its approval estimate. Supply a dated rate for the
selected account and machine.

`runtime.maximum_generated_candidates` is set by eight files and carried by twenty eight after profile resolution. Two set 2000:
`full-ensemble.template.json` and `small-run.template.json`. Six set 20:
`local-contract-test.json`, `modal-draft.json`,
`modal-first-run-floor.json`, `modal-smoke.json`,
`provider-canary.template.json`, and
`rfdiffusion3-two-arm-contract-test.json`. After resolution, nineteen carry 2000 and nine carry 20. Read the resolved profile, because an overlay can inherit this cap.

### Calculate the example fanout

The shipped estimator counts workload. Compose the demo campaign against the
Modal profile and read the fanout:

```bash
PYTHONPATH="$BINDER_SKILL" python3 -B -c 'import json,tempfile;from pathlib import Path;from claude_binder import lane;from claude_binder.paths import package_root;K=package_root();out=Path(tempfile.mkdtemp())/"composed.json";lane.compose_campaign(K/"data/demo/tslp/campaign.json",K/"data/templates/profiles/small-run-modal.template.json",out);print(json.dumps(lane.estimate_fanout(json.loads(out.read_text()))["counts"]["provider_calls_including_smokes"],indent=2))'
```

It returns 12 generated candidates, 14 screen predictions, 0 optimization predictions, and 70 rescore predictions. That is **84 co-folds** at `selection.final_count` 10, split 14 screen and 70 rescore. Optimization is disabled in this profile. These are planned counts, not settled charges.

#### What moves the co-fold count

`estimate_fanout` drives screening and rescoring from different counts:

```text
screen       = (generated + controls) * targets * predictors * screen_seeds
optimization = variants * targets * predictors * screen_seeds
rescore      = (rescored + controls) * targets * predictors * rescore_seeds
```

Each seed term is the number of distinct seeds in that list. `generated` is
`backbone_count` times `sequence_design.sequences_per_backbone` for each
compatible designer, plus `backbone_count` for each codesign generator.
`controls` includes enabled positive and negative controls. `variants` is
`optimization.rounds` times `parent_count_per_round` times
`variants_per_parent`, and is zero when optimization is disabled. `rescored`
is `selection.final_count` while optimization is disabled and
`parent_count_per_round + variants` once it is enabled. Smoke stages add
separate calls, one screen fold and five rescore folds in this profile.

- **One more generated candidate adds `targets * predictors * screen_seeds` screen folds.** In this profile that is one fold.
- **One more selected candidate adds `targets * predictors * rescore_seeds` rescore folds.** In this profile that is five folds, and in an optimization-enabled profile it is none.

Six folds is the price of raising both counts by one in this profile. Hold
`backbone_count` at 10 and raise `final_count` to 12 and the estimator returns
14 screen and 80 rescore, which is 94. Raise `backbone_count` to 12 as well and
it returns 16 and 80, which is 96. Raise `backbone_count` alone to 12 and it
returns 16 and 70, which is 86.

`selection_reachability` refuses the first case before dispatch because
`selection.final_count` cannot exceed the configured generation scale.
Generating more without selecting more remains valid. In
`full-ensemble.template.json`, raising that profile's `final_count` from 3 to
4 leaves the estimate at 1,125 co-folds, unchanged, because optimization is
enabled. `estimate_fanout` charges four screen folds before the first generated
candidate, three control folds and one smoke fold.

For this Modal profile, co-folds run `24 + g + 5 * f`, where `g` is generated
candidates and `f` is `selection.final_count`. The 24 is three controls at six
folds each plus six smoke folds. `claude_binder.cost` omits controls and smoke
calls. Applied once per phase to the composed Modal round of 10 candidates, at
one screen seed and five rescore seeds, it gives 10 screen folds plus 50
rescore folds, or 60. `lane.estimate_fanout` counts 84. The 24-fold gap is 18
control folds, three controls at six folds each, plus 6 smoke folds. Use the
graph estimate for admission control.

## Cost terms beyond predictions

Deploy-time cold starts, idle workers, storage, endpoint checks, failed
attempts, and teardown can bill outside a simple per-prediction calculation.
A retry is another attempt. Include a bounded allowance when these terms are
unknown and inspect the first selected-route receipt before scaling.

## Recorded route observations

The observations below refer to this package's recorded work. They do not
establish prices for a different deployment or target.

### Sequence design

The [fal billing observation](current-status.md#fal-billing-observations-2026-09-19)
combines an M-tier toolcheck and one sequence-design call at 0.0081026264 USD
settled. It does not isolate a per-design charge.

### Backbone generation

The package has qualified generation routes, but no transferable settled
per-backbone price is asserted here. Price the selected tool, route, and batch.

### Co-folding and screening

The demo Modal profile's 84 folds are a planned count. A completed 16-stage
supplied-candidate Modal campaign reserved 4.2837978 USD in admission
estimates under a 5 USD cap; Modal did not report a settled per-job amount for
those attempts. This is cap evidence, not a unit fold cost.

The separate [fal H100 toolcheck charge](current-status.md#fal-billing-observations-2026-09-19)
was 0.0843073825 USD settled. It does not price a prediction. Failed exports
and timeout attempts can consume provider time; budget retries.

### fal ESMFold2-Fast: the per-request model load is the cost

Eight 2026-09-04 requests on a Binder fal deployment included seven successful
HER2/affibody predictions and one failed request. The endpoint reported
348.656 to 435.391 seconds for successful requests, including 210.029 to
260.122 seconds of model loading. These are runtime observations, not settled
per-request bills. The application reloaded its model for each request; a
different deployment may batch requests or reuse its model. The failed
request returned HTTP 500 after about 1712 seconds and no structure. The
record does not establish why it failed.

### AF2-backprop design

[BindCraft2 on Modal](current-status.md#bindcraft2-on-modal-2026-09-21) recorded
a 17.3-second A100-80GB toolcheck, a 619.0-second upstream execution, and a
370.2-second repeat with a warm compilation cache. Setup on CPU took 100.6
seconds. The adapter parsed the output. Full Binder profile execution and its
downstream screen remain unverified; settled dollars are pending.

## Provider rates

Bundled rate records demonstrate the `--rate-record` interface. Obtain the
selected account's rate and quote date, hardware, region, and billing units.
Omitting `--rate-record` returns UNKNOWN.

## The gates a candidate must clear

Calibrate a target-matched positive control and shuffled-sequence negatives
for each chosen predictor. Fix seeds and ranking rules before screening. See
[target qualification](target-qualification.md).

## Two ways a round gets paid for twice

A novelty filter without its declared reference digest is refused by the
package before execution. A resumed run checks completed-stage artifacts and
bundle identity before replaying a stage; follow [resume](resume.md). Verify
control calibration file presence and digest before paid folds too.

## Where the spend ledger stands today

The guarded Modal dispatcher checks `enforce_spend_cap` and writes a
`charge-estimate` reservation before submission. Its key includes run, stage,
and attempt, so a retry reserves again. `close_wave` replaces the estimate
when attributable provider usage is available; otherwise the reservation
remains held against the cap.

One completed 16-stage supplied-candidate Modal campaign recorded 4.2837978
USD of admission estimates under a 5 USD ceiling and no settled per-job
amount. That is not a zero bill. A workspace billing report aggregates apps;
no shipped job-to-billing join establishes an attributable campaign total.

## The ceiling the code enforces today

Every paid stage on the guarded campaign path requires a positive
`provider.budget.maximum_spend_usd` and a plan-bound approval. A configured
account-policy cap can lower the effective ceiling. `enforce_spend_cap`
refuses a stage when cumulative charges or reservations plus the remaining
plan estimate exceed the cap. Its register is under `runtime.run_root`, so
**cumulative means within one run**. Track an account-wide ceiling across
separate runs outside that register.

The inner `python3 -m claude_binder.canary_runner run` command has no dollar
cap. Qualify adapters through `python3 -m claude_binder.qualify run`, which
presents an estimate and accepts `--confirm-cost` and `--max-cost-usd`.
The approval gate uses plan estimates; it does not poll provider balance APIs.
