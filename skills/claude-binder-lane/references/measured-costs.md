# Cost and spend

Price a campaign from its materialized workload and the selected provider's
rate. Claude Science records a ceiling before paid work and updates the estimate
from each job receipt. A design count alone omits controls, predictor arms,
seeds, setup, and retries.

## Make the cost decision

1. Count the selected stages, candidates, controls, predictor arms, seeds,
   smoke calls, and retries with `estimate_fanout` or the small-campaign plan.
2. Read the rate for the selected account, hardware, storage, region, and job
   lifecycle. Include an allowance for setup and uncertain runtime.
3. Present the provider, hardware, data destination, per-job estimate, and
   campaign ceiling. Save the scientist's approval with the plan.
4. Run one bounded handoff on a new route. Use its recorded start, end, and
   billable duration to revise the remaining estimate before scaling.

The guarded dispatcher checks `cumulative + remaining_estimate` before a paid
stage. **Cumulative means within one run**; separate runs need their own
accounting against an account-wide budget. The ceiling is an approval limit,
not a prediction of the final bill. [Approval and spend](approval-and-spend.md)
defines the plan-bound record.

## Cost kinds and provenance

| Kind | Meaning |
| --- | --- |
| Settled billed amount | The provider reported a charge attributable to one job. |
| Provider-reported rate | A quoted rate for hardware or a service; duration turns it into an estimate. |
| Modeled estimate | A plan or calculation that has not been reconciled to a bill. |
| Unpriced | Execution occurred without an attributable charge. |

Claude Science's frame-level `total_cost` records model-price telemetry. A
provider workspace total can include unrelated jobs. Keep both separate from
the campaign's per-job compute receipts.

## Recorded Binder observations

- A completed 16-stage supplied-candidate Modal campaign held **4.2837978 USD
  in admission estimates** under a 5 USD ceiling. It has no settled per-job
  amount, so the reservation is cap evidence rather than a unit price.
- A fal M-tier toolcheck plus one sequence-design call settled at
  **0.0081026264 USD**. A separate H100 toolcheck settled at
  **0.0843073825 USD**. Neither charge prices a full campaign. The
  [fal billing record](current-status.md#fal-billing-observations-2026-09-19)
  gives their scopes.
- A BindCraft2 Modal toolcheck and upstream run have recorded A100-80GB
  [durations](measured-costs-detail.md#af2-backprop-design). Their settled
  dollar charge is pending.
- The packaged Modal example plans **84 co-folds** for 10 selected candidates,
  including controls and smoke calls. The [fanout calculation](measured-costs-detail.md#what-moves-the-co-fold-count)
  derives that count from the profile.

These observations describe their recorded routes and inputs. Select the
current account rate and record new job durations for another campaign.

## Stage and control guidance

### Sequence design

Price each selected toolcheck and design batch on its route. The fal settled
observation above combines a toolcheck and one design call.

### Backbone generation

Generation setup and batch size vary by tool. Price the selected image, weights,
candidate count, and wall-time limit.

### Co-folding and screening

Count positive and negative controls, screen and rescore seeds, each predictor,
and smoke calls. The 84-fold example is a count, not a billed cost.

### AF2-backprop design

For BindCraft2, `number_of_final_designs` is a floor rather than a fixed job
length. Bound a shard by wall time and use the selected machine's rate. The
recorded warm and cold timings are in [cost details](measured-costs-detail.md#af2-backprop-design).

### fal ESMFold2-Fast: the per-request model load is the cost

Eight Binder fal requests included seven successful predictions and one failed
request. Successful requests reported 348.656 to 435.391 seconds, including
210.029 to 260.122 seconds loading the model. These durations have no matched
settled bill. A batched or warm deployment may have a different cost profile.

## Provider rates

Use a dated quote from the selected provider account. Record hardware, region,
billing unit, storage, startup behavior, and the rate source. A historical
rate file demonstrates the `--rate-record` interface; it does not quote a new
user's account.

## The gates a candidate must clear

Calibrate a target-matched positive control and shuffled-sequence negatives on
every chosen predictor. Fix the seeds and reduction rule before ranking. The
[target qualification](target-qualification.md) and [scoring](scoring.md)
references describe the control and metric contracts. A model score produces a
computational shortlist, not a binding result.

## Approval and accounting limits

The guarded campaign path requires a positive
`provider.budget.maximum_spend_usd` and a plan-bound approval. An account-policy
cap can lower the effective ceiling. A retry reserves another attempt. The
[ceiling mechanics](measured-costs-detail.md#the-ceiling-the-code-enforces-today)
and [ledger behavior](measured-costs-detail.md#where-the-spend-ledger-stands-today)
explain the calculation and receipt scope.

The inner `canary_runner run` command has no dollar gate. Use `qualify run` with
its cost confirmation and maximum for an adapter qualification.
