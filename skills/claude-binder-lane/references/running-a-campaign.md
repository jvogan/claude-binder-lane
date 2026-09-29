# Running a campaign

After `preflight` passes and an operator authorizes the plan, run `execute`.
The executor follows the materialized dependency graph, writes mutable state in
the run root, and keeps the run bundle unchanged. The displayed plan order does
not prevent a dependency-ready stage from completing before an unrelated stage.
Before each paid-stage command,
the executor checks the approval record and the spend cap, and it checks
provider authorization before the selected paid stages begin.

Read [approval and spend](approval-and-spend.md) again whenever the plan, the
provider access, or the ceiling changes.

## Check handoffs before scaling

Work backward from the planned scores to the required artifacts. Confirm the
selected deployment's request schema and output fields from its documentation
and recorded responses. A model name or a successful HTTP response does not
establish that a required matrix, structure, or alignment is present. Read
[co-fold scoring hazards](cofold-scoring-hazards.md) before adding a predictor.
Resolve a known output gap before generating a batch. Where evidence is missing,
use the first authorized sample to check the downstream parser too.

Resolve chain roles from the target sequence and residue map at every tool
boundary. A generator or predictor can rename chains. Record the new mapping
and verify that sequence design preserves the fixed target before scaling.
Read the selected endpoint's existing route notes before probing paths or
guessing request fields.

Use explicit working directories and resolved paths across cells. Inspect the
files a successful process actually wrote before parsing them. ProteinMPNN's
packaged [designer adapter](../claude_binder/adapters/proteinmpnn_designer.py)
provides `parse_fasta_records`, `header_designed_chains`, and `chain_segment`.
Use the native headers to distinguish the reference record from designed
samples and to identify sequence segments. An output containing one designed
chain may have no slash. Count parsed samples and preserve rejection reasons.

## Start with supplied candidates

When the campaign already has complete candidate records, select
`supplied-candidates-modal.template.json` for guarded Modal compute or
`supplied-candidates-fal.template.json` for the packaged fal endpoint. Both
profiles remove backbone generation and sequence design. Each validates one
candidate in the smoke phase, then processes the manifest's declared row count
in the scale phase.

Add this object to `campaign.json`:

```json
{
  "supplied_candidates": {
    "manifest_path": "inputs/candidate-manifest.jsonl"
  }
}
```

Each JSONL row follows `$BINDER_SKILL/claude_binder/data/schemas/candidate-manifest.schema.json`. Supply the
candidate ID, one-sequence FASTA, candidate structure, design pose, content
hashes, target and residue-map identities, and lineage fields. The design pose
is required because Binder measures each predicted complex against that
reference. Use a standalone Claude Science co-fold route for FASTA-only work
that does not request Binder's design-pose comparison metrics.

Compose with the supplied-candidate profile for the selected route:

```python
from claude_binder.paths import package_file

assert binder_cli([
    "compose",
    "--campaign", "campaign.json",
    "--profile", str(package_file(
        "data", "templates", "profiles",
        "supplied-candidates-modal.template.json",  # or supplied-candidates-fal.template.json
    )),
    "--out", "composed.json",
    "--json",
]) == 0
```

Materialization copies every supplied file into the immutable run bundle. It
verifies the declared hashes and rewrites the manifest paths to the bundled
copies. A missing file, changed hash, duplicate candidate ID, or normalized
path collision refuses materialization before provider work.

## Run the materialized plan

The `execute` example below is for local and other directly executable routes. For Modal, run the
reachable local stages individually, then use the guarded `dispatch_modal.py` workflow from the
Claude Science `repl` kernel. Direct Modal execution through `execute` is intentionally disabled.

Dry-run the plan first. A dry run renders the selected stage commands without
running them and without any provider work, and it still reports a failing
contract audit or control-separation gate.

1. Dry-run the full selected graph and require exit code 0:

   ```python
   assert binder_cli([
       "execute", "--plan", str(bundle / "run-plan.json"),
       "--stage", "all", "--dry-run", "--json",
   ]) == 0
   ```

2. Read the printed errors list. A non-zero exit or a non-empty errors list
   names the contract audit or control-separation finding to fix before the
   paid run.

3. On a directly executable route, once an operator approves the plan, run the same selected graph with a
   visible run root, then print the run state and the completed stage IDs:

   ```python
   import json
   from pathlib import Path

   run_root = Path("claude-binder-paid-run/run")
   assert binder_cli([
       "execute", "--plan", str(bundle / "run-plan.json"),
       "--run-root", str(run_root), "--stage", "all", "--json",
   ]) == 0

   status = json.loads((run_root / "status.json").read_text(encoding="utf-8"))
   print(status["state"], status["completed_stages"])
   ```

`--run-root` overrides the run root stored in the plan. Without it, `execute`
uses `plan["runtime"]["run_root"]`, and `materialize` renders that value with
its data root. `--data-root` takes precedence over `CLAUDE_BINDER_DATA_ROOT`.
The default data root is `.claude-binder` in the working directory. The
shipped small-run template sets its run root to `{{data_root}}/runs/{{run_id}}`.

## Read the run root

The executor writes these run-root files during a non-dry execution.

| Path | Meaning |
| --- | --- |
| `status.json` | State, completed-stage IDs, skipped-stage details, and provider-call summary. |
| `run-pointer.json` | Current stage, remaining-stage count, bundle path, and run-root path. |
| `run.lock` | Active-run lock. The executor removes it when the execution exits. |
| `runtime-config.resolved.json` | Runtime configuration with bundled input paths. |
| `stage-checkpoint.json` | Hash-verified completed-stage checkpoint used by `--resume`. |
| `partial-summary.json` | Failure or interruption summary with a resume command and action. |
| `keep-list.json` | Durable-handoff list with file status, hash, and byte count. |
| `artifacts/receipts/<stage-id>.json` | Receipt for each completed stage. |
| `artifacts/stage-progress.jsonl` | Stage events. |
| `artifacts/executed-commands.jsonl` | Commands that the executor launched. |
| `artifacts/spend.jsonl` | Estimated spend records for launched paid stages. |
| `artifacts/artifact-index.json` | Hash index written after execution reaches its terminal path. |

Raw stage files use `artifacts/stages/<stage-id>/attempts/<attempt-id>/<phase>/`.
The executor copies an output with a declared `publish_path` under `artifacts/`
only after the stage validates its output manifest.

Read `run-plan.json` for the selected profile's stage IDs, dependencies, and
declared published paths. The full-ensemble profile declares these common
published files:

- Normalized candidate rows: `artifacts/candidates/candidate-manifest.jsonl`.
- Screen scores: `artifacts/scores/screen-score-table.jsonl`.
- Promotion rows: `artifacts/promotion/promotion-manifest.jsonl`.
- Uniform rescore observations: `artifacts/scores/uniform-observations.jsonl`.
- Final ranked candidates: `artifacts/scores/ranked-candidates.json`.
- Control observations: `artifacts/controls/control-observations.jsonl`.

Sequence files remain in their declaring stage's attempt directory. The
candidate manifest and ranked-candidate rows include each sequence path and
SHA-256 digest. The keep list collects available FASTA files under
`artifacts/`, every `.pdb`, `.cif`, and `.mmcif` file under `artifacts/`, and
the design pose, predicted complex, and PAE matrix that each uniform
observation row names.

1. To print the top ranked sequences, run the package command:

   ```python
   binder_cli([
       "sequence-score-table",
       "--ranked", str(run_root / "artifacts" / "scores" / "ranked-candidates.json"),
       "--n", "10",
   ])
   ```

The browser viewer profile declares `artifacts/viewer/manifest.json`,
`artifacts/viewer/index.html`, and PNG files below
`artifacts/viewer/thumbnails/`. The full-ensemble profile can also declare
`artifacts/pictures/manifest.json`, `artifacts/pictures/index.html`, and
`artifacts/pictures/structure-pictures.zip`. These paths are profile-specific
declarations. Read [route status](current-status.md) before you
promise a provider-derived ranked result or picture.

## Follow optimization rounds

When `optimization.enabled` is true, `materialize` expands one stage graph for
each configured round. One `execute --stage all` follows those dependencies,
so you do not rerun the command between rounds.

Round 1 reads `artifacts/promotion/promotion-manifest.jsonl` and
`artifacts/scores/screen-score-table.jsonl`. A later round reads the preceding
round's `eligible-parents.jsonl` and `score-table.jsonl`. Each expanded round
publishes its summary, decision, optimized candidates, filter records,
predictor observations, score table, and eligible parents below
`artifacts/optimization/rounds/round-N/`. The final selection publishes
`artifacts/optimization/rescore-candidates.jsonl` for the rescore stages.

1. When a round stops, read the round's
   `optimization/rounds/round-N/next-round-decision.json`. A stopped decision
   has no selected parents, a candidate count of zero, and a non-empty
   `stop_reason`.

The registered outcomes are `continue`, `converged`, `all_metric_floor`,
`candidate_budget_exhausted`, `prediction_budget_exhausted`, and
`round_budget_spent`. A final round can carry `round_budget_spent` as its
termination outcome without setting `stop`, and `all_metric_floor` also
continues, marking a cohort whose primary metric is at its floor.

2. To answer why the loop ended, read
   `artifacts/optimization/decision-ledger.jsonl`. It holds one row per round
   with the round number, the outcome, the candidate count, and the
   `stop_reason` string. A convergence stop names the measured improvement and
   the configured margin, as in `round 1 improved ipsae_min by 0.01. The
   early_stop_margin is 0.029008`. A budget stop names the remaining slots and
   the planned candidates, as in `round 2 stopped because the optimization
   prediction budget has 0 remaining slots for 1 planned candidates`.

The optimizer carries the scored pool into a stopped round and records the
source optimization round for each carried row. A single tied comparison does
not establish convergence. Two consecutive ties can establish it when an
`early_stop_margin` is configured.

Changing `optimization.rounds` changes the plan and nothing else about the
campaign. Each round expands into six fixed stages, one plan, one optimize,
two filters, one measure, and one select, plus one cofold stage for each
enabled predictor. Against the three-predictor local contract profile that is
nine stages per round, so one-round, two-round, and three-round plans built
from the same campaign carry 35, 44, and 53 stages.

## Report completed work and cost

Report each arm's completed and remaining stages with its parsed output counts.
Generation, sequence design, and filtering can be complete while co-folding and
ranking remain pending. Keep attempted, emitted, accepted, and scored counts
separate so runtime and yield comparisons use the right denominator.

Report provider calls and costs per provider, including hosted inference.
An absent Modal submission says nothing about work performed by another
service. Keep measured duration, rate-based estimates, settled charges, and
unpriced calls separate. A cached image avoids a rebuild only when its identity
and required weights match; storage and later execution can still incur costs.
Read [measured costs](measured-costs.md) before revising the campaign estimate.

## Watch progress and recover

With `--json`, progress lines go to stderr and the JSON result goes to stdout.
Without `--json`, progress lines go to stdout. Each completed-stage line gives
the position, stage ID, result, elapsed seconds, output summary, and primary
published path when one exists.

1. Between cells, read the pointer's `current_stage_id` and
   `remaining_stage_count` from `run-pointer.json`. The executor rewrites
   `status.json` and `run-pointer.json` after every completed stage.
2. On failure or interruption, read `partial-summary.json`. It records
   `completed_stages`, `failed_stage`, `resume_command`, `resume_action`,
   `artifact_status`, and `claim_level: "blocked"`. `status.json` can report
   `failed`, `interrupted`, `completed_with_skips`, or `incomplete`.

An optional-stage failure produces a skipped receipt. A stage whose required
input belongs to a skipped dependency also receives a skipped receipt. A full
graph with skips cannot pass terminal validation as a clean result.

3. After you correct the reported cause, resume the same bundle and run root:

   ```python
   if binder_cli([
       "execute", "--plan", str(bundle / "run-plan.json"),
       "--run-root", str(run_root), "--stage", "all", "--resume", "--json",
   ]) != 0:
       raise RuntimeError("resume refused or failed; read partial-summary.json")
   ```

`--resume` verifies the materialized bundle identity and the completed
receipts, then reconciles valid receipts into `stage-checkpoint.json`. A selected stage may be
dependency-ready even when an unrelated earlier stage is not complete. A selected stage with an
unmet predecessor refuses before execution. If every stage is already checkpointed,
an unfiltered `execute --resume` runs local terminal validation without provider authorization
preflight or provider work. If either bundle identity or receipt validation fails, materialize a new bundle. Read
[resume](resume.md) for the recovery boundary.

## Preserve the results

The hosted workspace is deleted six hours after it goes idle. Promote files
after each artifact-writing stage and again before the idle deadline.

1. When `keep-list.json` exists, stage its available files for a flat
   handoff. The destination must be empty.

   ```python
   import json
   from pathlib import Path

   from claude_binder.artifact_handoff import stage_flat_artifacts

   keep_list = json.loads((run_root / "keep-list.json").read_text(encoding="utf-8"))
   available = [
       Path(item["absolute_path"])
       for item in keep_list["files"]
       if item["status"] == "available"
   ]
   staged = stage_flat_artifacts(
       available,
       source_root=run_root,
       destination=run_root / "flat-handoff",
   )
   ```

2. Promote the staged files with the bare kernel global `save_artifacts`. Do
   not use a `host.` prefix. The call accepts a file list positionally or as
   `files=[...]`, with optional `language=` and `version_of=` arguments.

   ```python
   save_artifacts(files=[str(path) for path in staged])
   ```

3. After a Modal `compute_done` notification, call
   `save_artifacts(payload["output_files"])` to promote the completed job. The
   payload already contains `output_files`, so you promote without re-entering
   the kernel.

CAUTION: `submit_job(outputs=[...])` decides which files return from `./out/`.
A file that a tool writes there is not kept unless the output list names it.
The run's own `stdout.log` and `stderr.log` always land as logs. An omitted
output can be lost when the workspace expires.

4. Keep a manifest of the intended name, byte count, and SHA-256 digest.

[Getting started](getting-started.md) records where a saved artifact is written, the
two retention values `destination` accepts, and the size limit that refuses a
save.

## Run specialized commands

The package includes five specialized subcommands for static contract verification, offline fixture testing, unconstrained contact clustering, post-run spend reconciliation, and optimization decision validation.

1. Audit a materialized plan against adapter implementations:

   ```python
   assert binder_cli([
       "contract-audit",
       "--config", str(bundle / "config.resolved.json"),
       "--plan", str(bundle / "run-plan.json"),
   ]) == 0
   ```

   The parser registers at `claude_binder/lane.py` with help string `"compare a materialized plan against the adapter code it will run"`. The command requires `--config` and `--plan`. Optional flags include `--package-root`, `--json`, and `--include-smells`. `contract-audit` compares a materialized run plan and resolved configuration against adapter source code to detect undeclared input paths, missing configuration keys, and mismatched output artifact schemas before execution. It launches no command and makes no network call, so run it before authorizing paid execution. `preflight` additionally checks selected-provider authorization and the spend cap, so it can correctly refuse a plan whose static audit is clean. Both commands take the materialized `config.resolved.json`, not `composed.json`.

2. Validate stage contracts against an offline fixture tree:

   ```python
   assert binder_cli([
       "contract-dry-run",
       "--plan", str(bundle / "run-plan.json"),
       "--artifact-root", str(completed_run_root / "artifacts"),
       "--from", "output-check",
   ]) == 0
   ```

   The parser registers at `claude_binder/lane.py` with help string `"validate later stage contracts against an offline fixture tree"`. The command requires `--plan` and `--from`. Optional flags are `--artifact-root` and `--fixture-root`. `contract-dry-run` links the resolved configuration and executes stage contracts from the selected starting stage onward against a local fixture tree with stubbed provider clients. Use it to develop or debug a downstream stage against intermediate mock artifacts, without running the upstream generation stages and without paying for provider compute.

   Use `--artifact-root` when the plan and the source artifacts belong to different runs. It points
   to the completed run's `artifacts/` directory. `--fixture-root` names an isolated output tree and
   must not overlap it. The supplied-candidate downstream route is covered through its viewer path.
   Alternate predictor, generator, and optimization routes do not receive a generic runner. A
   selected stage without an explicit runner is a reported failure. For a complete offline
   exercise of the full campaign graph, run the `local-contract-test.json` profile with
   `lane execute`; this profile runs every stage with fixture adapters and starts no
   provider job.

3. Cluster unconstrained screening contacts by surface interaction patches:

   ```python
   assert binder_cli([
       "discover-contacts",
       "--config", str(bundle / "config.resolved.json"),
       "--observations", str(run_root / "artifacts" / "scores" / "screen-score-table.jsonl"),
       "--out", str(run_root / "artifacts" / "discovery" / "contact-clusters.json"),
   ]) == 0
   ```

   The parser registers at `claude_binder/lane.py` with help string `"cluster unconstrained screening contacts without choosing a hotspot list"`. The command requires `--config`, `--observations`, and `--out`. The optional flag is `--json`. `discover-contacts` clusters scored screening candidate models by their exact contacted target-residue sets for targets configured with unconstrained discovery and writes the cluster summary to JSON. Run it after an unconstrained screening run to group candidate models by the target surface patch each one contacts. It selects no binding site for you.

4. Reconcile estimated stage spend with settled billing records:

   ```python
   assert binder_cli([
       "reconcile-spend",
       "--run-root", str(run_root),
       "--settled", "cofold-screen-esmfold2-fast=0.88",
       "--source", "Modal invoice INV-2026-08",
   ]) == 0
   ```

   The parser registers at `claude_binder/lane.py` with help string `"replace finished-run estimates with caller-supplied settled provider costs"`. The command requires `--run-root`, `--source`, and at least one repeated `--settled` stage cost. The optional flag is `--json`. `reconcile-spend` writes reconciliation charge entries into `artifacts/spend.jsonl`, replacing the modeled stage estimates with settled provider costs from a verified billing source. Run it once a paid campaign has finished and its billing has settled. [Measured costs](measured-costs.md) distinguishes a settled billed amount from a modeled estimate.

5. Validate a next-round optimization decision:

   ```python
   assert binder_cli([
       "validate-decision",
       "--config", str(bundle / "config.resolved.json"),
       "--round-summary", str(run_root / "artifacts" / "optimization" / "rounds" / "round-01" / "round-summary.json"),
       "--decision", str(run_root / "artifacts" / "optimization" / "rounds" / "round-01" / "next-round-decision.json"),
   ]) == 0
   ```

   The parser registers at `claude_binder/lane.py` with help string `"validate a bounded optimization decision"`. The command requires `--config`, `--round-summary`, and `--decision`. Optional flags include `--out`, `--ledger`, and `--json`. `validate-decision` checks a next-round optimization decision JSON file against campaign configuration rules and round summary metrics, covering parent eligibility, adapter operation permissions, parameter ranges, fanout limits, and stopping state consistency. Run it on an externally produced or hand-written decision before advancing to the next round.

## The full command surface

`binder_cli` registers nineteen subcommands. The walkthroughs above cover the
ones a campaign needs in order, and this table lists all of them with the help
string each parser declares. Run any of them with `--help` for its flags.

| Command | What it does |
| --- | --- |
| `version` | show package version |
| `check` | validate a campaign configuration |
| `discover-contacts` | cluster unconstrained screening contacts without choosing a hotspot list |
| `preflight` | audit a materialized plan before any stage starts |
| `link` | statically link composed declarations to adapter resolutions |
| `contract-audit` | compare a materialized plan against the adapter code it will run |
| `capabilities` | measure provider authorization and report available campaign shapes |
| `tools` | list every catalogued tool for planning without probing or dispatching |
| `contract-dry-run` | validate later stage contracts against an offline fixture tree |
| `compose` | combine a campaign with an execution profile |
| `qualify` | quote or run target-specific model canaries and write a model roster |
| `gate-scaffold` | write a target-bound schema-v4 FAIL gate ready for measured controls |
| `materialize` | write a resolved run packet |
| `execute` | execute materialized stage commands |
| `reconcile-spend` | replace finished-run estimates with caller-supplied settled provider costs |
| `rank` | aggregate observations and select candidates |
| `sequence-score-table` | print ranked sequences with the scores present in the ranked output |
| `merge-shards` | fold one sharded phase into the single output tree the stage contract names |
| `validate-decision` | validate a bounded optimization decision |

`execute` already ranks a completed run, so `rank` is what you call to aggregate
observations again without re-running a stage. `merge-shards` applies to a plan
that set a shard width, which is how [approval and spend](approval-and-spend.md)
keeps a large fanout inside its ceiling.

## Known limits

The live PD-L1 records cover cross-provider co-folding, ranking, a three-target
counter-screen, three-seed follow-up, and one supplied-candidate Modal campaign
through all 16 stages. The guarded `dispatch_modal.py` path completed smoke,
bounded scale, artifact return, local continuation, promotion, and
`render-viewer` in Claude Science. A read-only replay validates the completed
run's 16 receipts and artifacts. The provider supplied no per-job settled
amounts, so the actual Modal bill remains unknown.

Direct paid Modal stages through `claude-binder execute` remain disabled. That
synchronous surface cannot receive the completion notification that Claude
Science delivers between cells. Use the guarded Modal wave dispatcher for paid
Modal stages. Alternate generator, predictor, and optimization routes remain
coverage gaps until each selected route has its own receipt-backed canary.

The fal route has since completed a paid campaign end to end, 16 of 16 stages
committed through the packaged client. An earlier Claude Science request
reached the selected application and returned no result before the package's
requested wall bound, which the package records as unknown remote state. The
1700-second request maximum is the value this package sends by default, not a
published fal platform limit, and it sets no sequence-length ceiling. Reconcile
a call left in that state before retrying it; unrelated routes and candidates
remain available.

The contract audit that `lane.py` runs before `target-prepare` takes about six
seconds at 211 MB.

Report a ranked design or promoted visual only when its provider record contains
the corresponding inputs, outputs, scores, and provenance. A computational
record does not establish binding or biological activity.

Source: [`lane.py`](../claude_binder/lane.py) defines execution,
materialization, run files, publication, resume verification, and progress.
[`artifact_handoff.py`](../claude_binder/artifact_handoff.py) defines
flat handoff. [`optimization_controller.py`](../claude_binder/adapters/optimization_controller.py)
defines round decisions and stopping behavior.
