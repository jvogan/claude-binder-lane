# Compute and pacing

Read this before the first dispatch, then at every budget cycle.

## Contents

- [The ceiling this install actually has](#the-ceiling-this-install-actually-has)
- [Sharing an account](#sharing-an-account)
- [The governor and the dispatch gate](#the-governor-and-the-dispatch-gate)
- [Pacing the ceiling](#pacing-the-ceiling)
- [The submit call, verified against the shipped compute skill](#the-submit-call-verified-against-the-shipped-compute-skill)
- [Job settings](#job-settings)
- [Volumes and ledgers](#volumes-and-ledgers)

## The ceiling this install actually has

Check this before you plan a wave. The published campaigns ran at a governor ceiling
of 325 concurrent GPU instances on a multi-target run and 150 on a single-target run.
Plan a stock Claude Science install against the provider cap below, not against the
published ceilings.

The compute provider carries its own per-provider concurrency cap, separate from the
session limit you set with `host.compute.set_concurrency_limit`. For a Modal provider
row with no explicit value, that cap defaults to **10 concurrent jobs**. The provider
enforces this limit at submit. Job 11 is refused with
`error_kind='provider_concurrency_full'`. A governor ceiling of 325 does nothing
until the provider cap is raised.

**Provider limit control**. The user raises the cap themselves. The field is
**Concurrent jobs**, in Settings under Compute on the Modal provider page, in the
"Spend & runtime" section. It defaults to 10 and accepts up to **10,000**. Their own
Modal plan's concurrency limit still applies on top, which the field's own helper text
says, so setting it high is a request to Claude Science rather than a grant from Modal.

Both published ceilings are inside that range. A run at 325 or at 150 needs the user to
set this one field, not a platform exception, so there is nothing here to ask a
platform owner for.

Do not read the SSH/BYOC error as a block on Modal. The daemon does guard
per-provider limits, and the guard's text reads "Per-provider limits are only
configurable for SSH/BYOC". It rejects the `local`, `proxy`, and `infer` families.
Modal's family is `byoc`, so the guard never fires for it.

Read the live pair back from `host.compute.status()`, which returns both the session
limit and the provider ceiling. Plan the wave against what you read there, and ask the
user to raise the field when the campaign needs more than it reports.

Then tell the campaign what the user set. `dispatch_modal.py` sizes each scale
subwave from **`provider.maximum_concurrent_jobs`** in the campaign config. Set it to
the value on the user's Modal row. Omit it and the dispatcher falls back to 10, the
platform's default for a row nobody has changed, so a campaign that omits the field
runs 10-job subwaves. Do not set it above the row's value: the provider enforces
concurrency at submit, and the surplus jobs of an oversized wave come back
`provider_concurrency_full`.

No daily compute-hours budget applies to Modal. Provider-backed reservations pass an
unbounded hours-per-day. The platform's 40 compute-hours-per-day budget and its default
of 5 concurrent containers belong to the local metered tier, so neither bounds a
campaign running on Modal. Modal spend is bounded by the ceiling in
`provider.budget.maximum_spend_usd` and by the pacing rules below, and by nothing else.

Verified on Claude Science 0.1.41-release by reading the daemon's guard, its
`ops.compute.updateProviderDetails` entry, and the settings control that calls it.

## Sharing an account

A shared account also holds other campaigns' volumes, apps, running sandboxes, threads and
folders. Do not read, write, delete, terminate or post into any of them, and do not treat
their existence as an anomaly to reconcile. A volume that already exists on the account
belongs to someone else.

The governor and the dispatch gate count live GPU sandboxes carrying this campaign's own
project tag and nothing else. Load on the account outside that tag is expected, and it is
never a reason to throttle, halt, or raise an alarm.

Do not write to `compute_details` for `byoc:modal`. Record environment findings as
project artifacts instead. Expect to **read** other people's notes there. The document
is one per provider, shared across every session and project on the install, and it is
already populated on a machine that has run any earlier Modal work. An existing
`### env:` block in it is not yours. Its cap is 32,768 bytes. An append that exceeds the cap causes an error.

## The governor and the dispatch gate

One campaign-wide integer, the **governor ceiling**, bounds concurrent GPU instances. Record it
every cycle with its basis, the metered and rate-card spend, the live sandbox count, and the
elapsed and spent percentages.

The opening ceiling holds until spend instrumentation is validated against metered billing and
every model in the funnel has a per-design wall-clock benchmark.

Check the ceiling from one shared code path, **immediately before each dispatch**, and read it
fresh each time. The check is fail-closed: an error inside it refuses the dispatch. A delayed
dispatch costs seconds and an ungated GPU dispatch costs unrecoverable dollars.

Refuse the dispatch when any of these hold.

- The ceiling reading is missing, or was last set more than 25 minutes ago.
- The ceiling is zero.
- The requested timeout exceeds 1800 seconds while spend is more than 5 points over elapsed.
- The live GPU sandbox count is within `max(10, floor(0.1 * ceiling))` of the ceiling. Back
  off with jitter and stop retrying after eight attempts.
- The target's generated count has run ahead of its scored count past the campaign's
  backlog cap.

  The backlog cap carries no recorded value here. Name the value before the first
  dispatch.

Count live sandboxes through the provider SDK, all sandboxes minus the CPU utility sandboxes
you tagged, never through a usage table. Tag every CPU utility sandbox and record its ID so
the count subtracts it.

A per-track cap is an advisory subdivision of the one ceiling, never additive to it.

## Pacing the ceiling

Raise the ceiling only on a **metered** spend reading you trust, never on a rate-card
reconstruction. Lowering, holding, and writing zero act on whichever of metered and projected
is higher.

| Rule | Value |
|---|---|
| Step up per budget cycle | at most 50 instances |
| Pause new dispatch | spend at or above elapsed + 15 points for one cycle, write ceiling zero |
| Scale up | spend below elapsed - 20 points for three consecutive cycles |
| Lower | live above ceiling for two consecutive cycles, drop to 80% of current |
| Never raise while | two metered readings disagree by more than 20% |
| Never raise while | rate-card exceeds metered by more than 50% and more than $200 |
| Never raise while | live exceeds the current ceiling |

Writing ceiling zero pauses **new** dispatch only. In-flight jobs are never cancelled on a
budget signal.

The terminal-sandbox reap sweep runs **every cycle**, and a skipped sweep is a logged
deviation. A dispatch-gate block that lasts two cycles while pace is behind triggers a reap
and a recount before the block stands.

Any throttle, cap, or pause carries an explicit expiry or review checkpoint and applies to new
dispatch only. Lift it within two hours of the condition that prompted it clearing.

Billing lags. Force the meter early: dispatch a tiny CPU-only probe under 60 seconds as one of
the first three jobs so the account starts billing, and record a short canary job. Until a
calibrated reading exists, every posted figure is labeled uncalibrated and no pace adjective
accompanies it. A reading is calibrated only when the canary's cost appears, the value clears
a floor, it differs from the previous reading, and its timestamp has advanced.

If the canary has not appeared in the metered report 90 minutes after start, post a one-line
billing-dark warning. If no calibrated reading exists by three hours, stop raising the
ceiling, hold it where it is, and repeat the notice each hour until one arrives. Lowering
stays available throughout.

## The submit call, verified against the shipped compute skill

Read the compute API from the installed skill in the org tree, never from a cached copy. Five
fields differ between a stale cache and what ships, and code written from the stale form fails
at the first submit.

| Field | The shape that works |
|---|---|
| Provider argument | `host.compute.create('modal', ...)`. The database row reads `byoc:modal`; that string is not the create argument |
| `provider_params` | One flat dict: `image`, `env`, `gpu`, `cpu`, `memory`, `volumes`, `timeout` |
| Job guard | `run_timeout_s` |
| Input records | `dst` is a bare filename. A `dst` containing a directory is refused at submit |
| Notification payload | `state`, `output_files`, `exit_code`, `notes` |

`outputs=` is a plain list of globs with an `exclude=` companion, and there is no per-glob
visibility record. Inputs cap at two limits per submit, a total size and **64 files**, and the
file count is refused before anything stages.

One notification return carries a list that may hold several completed jobs, each already
naming its state, exit code, and output files. A single wait drains every finished job.
Re-attaching reads the persisted result.

## Job settings

- Run every job on one GPU tier. A single tier removes the per-method tier-selection step and
  the mixed-fleet weighted-rate calculation.
- Set every job's timeout to `min(ceil(2.0 * benchmarked_p90_s), 5400)` seconds. Lower the per-job batch size when execution time approaches the limit. The provider enforces the value you pass by ending the container, so per-sandbox
  spend stays bounded even when every control loop is down.

  **5,400 seconds is this campaign's pacing choice, not a platform ceiling.** The
  platform ceiling on a Modal container is **85,500 seconds**, which is Modal's 24 hour
  sandbox lifetime minus a 600 second harvest margin and a 300 second teardown grace.
  The "Default container timeout" value in Settings is a default for jobs that omit
  `timeout`, and passing `provider_params['timeout']` overrides it up to that ceiling.
  Raise the per-job value on a deliberate decision about spend, never by default.
- Set a **300-second** idle timeout on every sandbox.
- End every job command with `mkdir -p out && cp -r <results> out/`, or write directly under
  `./out/`. A job that harvested two files or fewer is classified as a canary failure.
- Precede any fan-out above **10 jobs** with a single canary from the same spec, confirmed
  past startup with sane output.
- Terminate a harvested sandbox in the same cycle it was harvested.

Per-design wall clock and cost are not published for any campaign. Dividing the
published window and budget by the design range gives a campaign average across all methods.
The governor may not rise until every model in the funnel carries its own
benchmark. Benchmark each model in your own funnel before quoting a duration or a price, and
label every unbenchmarked figure an estimate.

## Volumes and ledgers

Batch per-design and per-seed outputs into one file per job: one metrics file and one
structures archive. A five-seed, million-design campaign exceeds a Modal volume's inode cap at
one file per design and seed.

Write the design-count ledger as **one file per writing frame**, never one shared file.
Appends to a shared file across concurrent sandboxes are not atomic and clobber silently. One
row per job completion, one writer per subfile, idempotent on the job and stage. Every
campaign-level total is a single aggregation over that ledger, never a count of output files.
A ledger that is still empty after scoring has completed means the scoring rows were
never written, so treat the stage as failed rather than as a zero.

The row schema is fixed: timestamp, job ID, target, structure method, stage, generated count,
scored count, GPU seconds, and the writing frame's ID. Everything a status post or a scoreboard
reports is an aggregation over those fields.

The reporting vocabulary is fixed at kickoff too. A design is **generated** when a generation
or sequence-design job writes its sequence. **Designs screened** is the sum of scored counts
over rows at the one-seed screen stage. **Designs ranked** is the sum over rows at the final
stage. Intermediate five-seed tiers carry their own stage value and count toward neither. Any
redefinition mid-campaign is a logged deviation carrying a one-time reconciliation row. Where a
run uses its own stage names, have each one declare which of these three it rolls up to.

Never move or rewrite a shared directory or file while jobs are writing into it. Route every
append to a shared ledger through its owning singleton after a fresh re-mount. Any delete on a
shared volume requires a verified archive first; if that step fails or times out, halt.

Give every volume a campaign-unique name that includes the first eight characters of the
root frame ID of the session that runs the campaign. A date-only suffix is not enough,
because another campaign on the same account can produce the same name on the same day.
Create the campaign volumes once, from a single writer. The published campaign uses four,
for state, the ledger, outputs, and the novelty corpus.

Prebuilt images and weight volumes are reusable infrastructure, re-validated against the PASS
criteria. Ledgers, novelty corpora, design outputs, and progress notes are created fresh per
campaign.

Use **exactly one Modal app** for the campaign, the session app Claude Science creates. Do not
create or deploy an additional named app. A Modal workspace holds at most 1000 apps, and reaching
that cap blocks new app creation across the whole workspace, including this campaign's own session
app.
