# Small campaign fast path

The installed `scripts/small_campaign.py` plans, admits, runs workers, scores and reports one bounded campaign.
Claude Science uses the scientist's provider account and joins real provider and worker receipts.

For a no-spend plan, read this page and only the target, selected tool, and selected route references needed to settle inputs; read build detail during execution.

## 1. Stage and plan

Copy `scripts/small-campaign-settings.template.json` to `settings.json`. Set target sequence, structure and MSA;
BindCraft2 settings and accepted CIF glob; provider, H100, egress hosts, absolute output path, and **the
user's** total and per-job USD ceilings. Stage inputs at absolute paths that will exist inside each worker, such
as `/work/campaign`. Warm public weights in a provider Volume; record image, weight digest and reuse in provider
receipts. A placeholder ceiling is never approval. Aim for 24–60 accepted designs, with one candidate carried
through the first complete handoff.

Choose `esmfold2-kit` plus `boltz2-kit` for two predictor lineages. Stock `esmfold2-platform` can replace the
kit, but is the same ESMFold2 lineage. The kit requires `prepare_wheels()` in staged CPU sandboxes, then
`build_env('esmfold2_kit_gpu', path=..., hydrate=True)`, an H100 `CHECK`, and `pred`:
[recipe](esmfold2-kit-modal.md). Stage its path and digest with
`binder_stage_environment_recipe('esmfold2_kit_gpu', workspace)`. [Accelerated
builds](accelerated-upstream-builds.md) covers BindCraft2 and Boltz-2.
The template selects `fast` for both kits, which engages acceleration and can change numerics.
Use `exact` only after checking fidelity for the chosen configuration, or `off` for a
deliberate stock comparison. Calibrate control thresholds separately for each selected
mode, target, and model revision; the template's PD-L1 rules are starting points.
The template uses five rescore seeds. For a custom campaign, Claude Science can
set a different list in `rescore.seeds`, then plan the resulting predictor calls
and measure controls with that same list. Record the selected seed count in the
report. The commands in this page use seed `0`, the template's first seed. If
you change `rescore.seeds`, replace `0` in the smoke and indexing commands with
the first seed in your list, and run the full jobs for every listed seed.
The candidate gate uses per-arm mean ipSAE and the maximum pose RMSD across
seeds. The template ranks by the mean of two control-normalized arm scores.
For the PD-L1 video example's headline score, set
`rescore.ranking_rule` to `mean_of_arm_best_ipsae`: it ranks passed candidates
by the raw mean of each arm's best ipSAE, without changing the control gate.

```sh
SC=/path/to/installed/skill/scripts/small_campaign.py
python "$SC" plan --settings settings.json --out design-plan.json
python "$SC" bundle-worker --out small-campaign-worker.zip
```

The bundle contains `small_campaign.py`, `small_campaign_esmfold.py`, and their
sibling `claude_binder` package. Record the printed SHA-256. In each Modal
`submit_job`, include `small-campaign-worker.zip` in `inputs` along with that
job's plan, approval, ticket, and other required input files. Mount the
selected image's weight Volumes as its recipe requires. The installed `SC`
path exists in Claude Science's local session; it does not exist inside a
fresh Modal image. Unpack the uploaded bundle into `/work/campaign/worker`
inside the job, verify it, and run its worker script there:

```sh
python -m zipfile -e small-campaign-worker.zip /work/campaign/worker
WORKER_SC=/work/campaign/worker/scripts/small_campaign.py
python "$WORKER_SC" verify-worker --root /work/campaign/worker \
  --archive small-campaign-worker.zip --expected-sha256 '<printed SHA-256>'
```

Run these three commands in the same admitted job as `run-design` or
`run-rescore`; the bundle check does not create another provider job. Give
the job an input mapping that places the target, MSA, BindCraft settings,
plan, approval, ticket, and the prior smoke receipt at the absolute paths
recorded in the plan and command. Keep the bundle digest and image identity
in the provider receipt. Test the command and file placement with a
one-candidate admitted job before scaling.

Review plan hash, candidate count, first-job estimate, provider, hardware, egress, destination and total ceiling
with the scientist. Obtain an operator-origin `design-approval.json` with `decision:"approved"`, exact
`plan_sha256`, `campaign_authorization_id`, provider/hardware/egress/destination, `maximum_spend_usd`,
`maximum_job_estimate_usd`, `approved_by`, `approval_ref` and `approved_at_utc`. The CLI checks these and source
hashes. Use one `admissions.jsonl` ledger and one authorization ID across design and rescore.

## 2. Admit design, then make the roster

Run `admit` immediately before each paid provider creation. It reserves the estimate under the remaining
campaign ceiling. Submit only with its ticket. Run the worker command inside the admitted job with the
verified staged bundle and inputs. A bounded first design job must yield an accepted pose before scaling. A failed
attempt consumes its reservation; retry with a new job ref.

```sh
python "$SC" admit --plan design-plan.json --approval design-approval.json \
  --ledger admissions.jsonl --job-ref design-0 --stage design --seed 0 \
  --estimate-usd "$DESIGN_ESTIMATE" --out design-0.ticket.json
# Inside the admitted worker, after extracting and verifying the bundle above:
python /work/campaign/worker/scripts/small_campaign.py run-design \
  --plan design-plan.json --approval design-approval.json \
  --ticket design-0.ticket.json --job-ref design-0 --seed 0 --out /work/out/design-0
```

Collect the accepted CIFs, `3_Ranked/!_Ranked.csv`, and worker receipt. Write `controls.json` as rows with
`name`, exact target/binder sequences and `kind`: one `positive_control` known partner ectodomain and shuffled
`negative_control` rows. IDs must match settings. Derive design chain IDs from CIF sequences. Use
`--target-state-suffix _stateName` if one ranked design has multiple state CIFs. Use the actual ranked path:

```sh
python "$SC" make-roster --settings settings.json \
  --ranked-csv '/work/out/design-0/accepted/3_Ranked/!_Ranked.csv' \
  --controls controls.json --design-target-chain A --design-binder-chain B \
  --limit 60 --out jobs.json
python "$SC" plan --settings settings.json --jobs jobs.json --out rescore-plan.json
```

The roster-bound plan records the selected seed IDs, full rescore jobs
(`predictors × seeds`), and planned prediction calls (`roster entries ×
predictors × seeds`). Roster entries include controls. Use these counts for
the provider estimate and campaign ceiling. For example, 45 candidates and
9 controls with two predictors and five seeds plan 10 full jobs and 540
predictions.

Approve this roster-bound plan with the **same authorization ID and no higher ceiling** in
`rescore-approval.json`. Do not treat the second plan as new spend.

## 3. Prove controls and one candidate in each arm

For each arm, admit a seed-0 smoke job and run `run-rescore --smoke`. It includes the positive, **all
negatives**, and one candidate; it parses their CIF and PAE outputs before scale. Example for ESMFold2 kit:

```sh
python "$SC" admit --plan rescore-plan.json --approval rescore-approval.json \
  --ledger admissions.jsonl --job-ref esm-smoke --stage rescore --arm esmfold2-kit \
  --seed 0 --estimate-usd "$RESCORE_ESTIMATE" --out esm-smoke.ticket.json
# Inside the admitted H100 worker with prepared kit and verified staged bundle:
python /work/campaign/worker/scripts/small_campaign.py run-rescore \
  --plan rescore-plan.json --approval rescore-approval.json \
  --ticket esm-smoke.ticket.json --job-ref esm-smoke --arm esmfold2-kit \
  --seed 0 --smoke --kit-root /kit/esmfold2 --out /work/out/esm-smoke
```

Repeat with `boltz2-kit` and `/kit/boltz2`. Confirm control separation, chain mapping and PAE orientation.
For each full job in the declared seed list, admit a fresh reference and run `run-rescore` with
`--smoke-receipt` for that arm.
Keep each job output separate. Provider receipts must include actual `provider_job_id`, `job_ref`, plan hash,
status/exit code, UTC start/end, wall and billable seconds, USD/hour, settled USD when available,
`worker_receipt_path` and SHA-256, and `admission_ticket_path`. Save design and full-rescore receipt arrays
separately.

## 4. Index, score and report

Fetch or tar large Modal results; filter bulk notifications while retaining job IDs and artifact hashes. For
each **full** job, inspect CIF/PAE and index that job's unique files. Derive chain order from sequences. Verify
PAE orientation from the emitted format: changing it shifted ipSAE_min by up to 0.11 in this workflow. Never
infer it from a filename. The kit scored about 0.2 above platform ESMFold2-Fast on PD-L1; their scales remain
separate.

```sh
python "$SC" index-artifacts --plan rescore-plan.json --arm esmfold2-kit \
  --seed 0 --job-ref esm-0 --prediction-root /work/out/esm-0/predictions \
  --target-chain A --binder-chain B --reference-target-chain A \
  --reference-binder-chain B --pae-orientation aligned_rows --out manifests/esm-0.json
python "$SC" score --predictions manifests/*.json --out scores.json
```

The manifests must cover all roster items in both arms and every declared seed. Save `design-link.json` as
`{"plan":"design-plan.json","approval":"design-approval.json", "receipts":"design-provider-receipts.json"}`.
Then:

```sh
python "$SC" report --plan rescore-plan.json --approval rescore-approval.json \
  --scores scores.json --receipts rescore-provider-receipts.json \
  --linked-phase design-link.json --out report.json
```

Read `report.json`, ranked `report.csv`, passed `report.fasta` and `report.txt`. The gate requires all controls,
two lineages, every declared seed, and binder C-alpha pose RMSD <=3.5 Å after target superposition. The
template's five-seed PD-L1 starting rules are: ESMFold2 kit mean-of-five >3× the largest shuffled negative's
best seed; Boltz-2 mean-of-five > the largest negative mean. Recalibrate on this target.
The report shows positive control, settled cost or pending
status, list-rate estimate, elapsed time and summed job duration. Missing artifacts, bad controls or absent
receipts stop it; fix the source issue and admit a new job for a retry.
