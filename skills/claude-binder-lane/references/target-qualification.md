# Qualify a target-specific model roster

Use this procedure before running an enabled model arm against a target that lacks a PASS roster row for the same target bytes. The qualification canary makes paid provider calls after cost confirmation.

A Modal arm takes a different route. The canary here runs through `subprocess`, and a Modal job needs the Claude Science kernel `host` object no subprocess holds, so a frame runs the canary and `lane qualify --receipt` turns its harvested receipt into a row. See [Qualify a Modal arm](modal-qualification.md).

The packaged roster records an unshipped qualification target using `UNVERIFIABLE_TARGET_NOT_SHIPPED` and stores the historical digest in `unshipped_qualification_target_structure_sha256`. Runtime validation refuses these rows for a materialized run. Requalify every enabled arm before using the roster for any target.

## Required inputs

Prepare these files before composing the campaign:

- A target structure file. The materializer hashes these bytes and the qualifier writes that digest to every roster row.
- The target sequence in `targets[0].target_sequence` or a FASTA path in `targets[0].target_sequence_path`. Providing an explicit sequence defines the intended target.
- A residue map at `targets[0].site.residue_map_path`. `materialize` hashes and bundles this file. The site definition references residues through this map.
- A positive-control complex for the same target and site from a literature structure or experimentally determined complex. Gate condition (b) requires its measured scores.
- A non-empty negative-control set. Gate condition (b) requires measured negative scores. [Build a negative control](#build-a-negative-control) writes one from a complex you already have, and states what each kind does and does not establish.
- One explicit `qualification.command_argv` entry per enabled model adapter. The command must write the adapter receipt and output files under the evidence root. The qualifier refuses production-stage command templates because stage commands can require artifacts unavailable to a standalone canary.

The target structure and residue map are required for `check --check-paths` and `materialize`. The positive-control complex and negative-control set are required for a PASS production scoring gate.

## Write the gate and drive it to PASS

Production scoring reads the gate directory and the stored status is fail-closed. Six steps take a target from no gate to a `PASS`.

1. **Create `gates/<target-id>.json` from the materialized target.** Do not copy the packaged `pdl1.json`. That record is `UNVERIFIABLE` because this package does not ship the 4ZQK-derived target and residue-map artifacts that would bind its PD-L1 control-panel evidence to a target dossier.
2. **Run `gate-scaffold`.** The command appears under [Commands](#commands). It writes a target-bound schema-v4 gate with every configured ranking arm present and every measurement left visibly required. It validates as `FAIL` by design, refuses overwrite, and authorizes no scoring until recorded measurements replace the required values.
3. **Set `runtime.validation_gate_dir`** in the qualification configuration to the absolute path of the `gates` directory.
4. **Record a measurement for every configured ranking arm.** Take the positive control for your target from the control table in [Published targets](published-targets.md), and build the negatives with [Build a negative control](#build-a-negative-control).
5. **Fill the `target_dossier`.** A `PASS` gate carries status `VERIFIED`, the target `source_id`, the materialized target-structure SHA-256, the design-target chains, the required entities, and the materialized site contract including its residue-map SHA-256.
6. **Meet both conditions.** The gate accepts `PASS` only when condition (a) recapitulates the target fold and condition (b) records `min_positive > max_negative` for `ipsae_min`.

The canonical `dossier_sha256` binds every recorded control-panel result in that gate to that exact construct and epitope. `gate_id` only selects a file. It never transfers evidence to a different target or site, which is why a gate is measured once per target per instrument and no threshold carries across.

The stored gate status remains fail-closed. Read each returned
`instrument_results[].qualification_status` before describing why. `NOT_MEASURED`
means the arm lacks one or more required measurements; it does not mean the
model failed a measured test. `MEASURED_FAIL` means a completed measurement
missed the declared condition. The output names unmeasured conditions so the
session can request or calculate only the missing evidence.

## Target confidentiality and alignment egress

Qualify the target's confidentiality before choosing an alignment route. A private or unpublished target sequence must not reach a third-party server. This section covers alignment egress. It does not cover the other paths that can carry target bytes off the machine.

Genie3's upstream binder preparation can call the ColabFold MSA server. Use local or precomputed MSAs for private or unpublished targets. The minimal installation described here omits ColabFold and the other upstream setup dependencies. The packaged adapter invokes `genie3 generate` and provides no `genie3 run` route, so it does not make that server call. An installation using upstream setup can include ColabFold; apply the same confidentiality requirement to its alignment route.

This lane's own alignment stage is where the decision is made. `claude_binder.adapters.target_msa_builder` builds one unpaired target-chain alignment per target, and its `--route` has no default because the route is a decision about network egress.

| Route | Egress | Satisfies the constraint |
| --- | --- | --- |
| `query-only`, through `--source` | None. Writes the query sequence alone. | Yes |
| `precomputed`, which `--source supplied-a3m` maps to | None. Stages an alignment file you already hold, named with `--precomputed-a3m TARGET_ID=PATH`. | Yes |
| `local` | None. An MMseqs2 database on your own machine. | Yes in principle. The adapter refuses this route for want of a database path and an execution implementation. |
| `public-server` | Sends the target protein sequence as FASTA with `mode=env` to `https://api.colabfold.com`. | No |

The public route already requires explicit consent. `build_alignment` raises `SourceUnavailable` unless `--allow-public-msa` is passed, and the refusal names the host. That flag guards against an accidental send. It records no judgement about whether the target may be sent.

One shipped profile gives that consent in advance. `full-ensemble.template.json` carries `--route public-server --allow-public-msa` in its `stage-msa` command template, so a confidential target run under that profile sends its sequence to the ColabFold host with no further prompt. Change the stage to `--route precomputed` and supply an alignment before running such a target through it.

Four other shipped profiles stage an MSA, and each names a source rather than a route. `modal-first-run-floor.json` and `small-run.template.json` pass `--source query-only`, which sends nothing. `modal-draft.json` and `modal-smoke.json` pass `--source colabfold-server`, the legacy name that `resolve_alignment_route` maps to `public-server`, and neither carries `--allow-public-msa`, so the stage refuses instead of sending. The consent guard produces that refusal, and it records no decision about whether the target is confidential. Set the source these two profiles want before a confidential target reaches them.

Record the decision. The route reaches the MSA manifest, so a scored row traces back to how its alignment was built.

## Build a negative control

A campaign requires a negative control unless you disable the control panel. `controls.minimum_negative_controls` defaults to 1, and gate condition (b) has nothing to compare against without one. [Run with no panel at all](#run-with-no-panel-at-all) gives the five fields that lower the minimum to 0, and what the run gives up in exchange. `claude_binder.make_negative_control` writes one from a complex you already have, along with the `controls.negative[]` block that declares it.

Run it before `compose`. It writes a structure file and a `.control-block.json` beside it. Paste the block into `controls.negative`.

### A sequence decoy, from one complex

This keeps the target chain and the binder backbone and permutes the binder's residue identities. Use it when the only complex you hold is the one your positive control uses. The source may be PDB or mmCIF, and the decoy is written in the same format.

```bash
PYTHONPATH="$BINDER_SKILL" python3 -m claude_binder.make_negative_control sequence-decoy \
  --structure "$CAMPAIGN_DIR/structures/positive-complex.cif" \
  --target-chain B \
  --binder-chain A \
  --shuffle-seed 23 \
  --control-id my-sequence-decoy \
  --campaign "$CAMPAIGN_DIR/campaign.json" \
  --out "$CAMPAIGN_DIR/structures/my-sequence-decoy.cif"
```

The seed makes the permutation reproducible. One seed and one source always write one file. The permutation is the one `adapters/control_builder.py` would have constructed in the run. The two files are byte-identical when the negative declares the chain IDs its source already carries, which is the shape this command produces. Under a chain rename the sequences still match and the bytes differ, because the in-stage branch renames chains and this command does not.

### A cross-pair mismatch, from two complexes

This takes the target chain from one complex and the binder chain from a different complex. Use it when you hold two complexes whose binders differ. Both sources must be PDB, because mmCIF splicing is unresolved.

```bash
PYTHONPATH="$BINDER_SKILL" python3 -m claude_binder.make_negative_control cross-pair \
  --target-structure "$CAMPAIGN_DIR/structures/target-complex.pdb" \
  --target-source-chain A \
  --binder-structure "$CAMPAIGN_DIR/structures/other-complex.pdb" \
  --binder-source-chain C \
  --target-chain A \
  --binder-chain B \
  --control-id my-wrong-pair \
  --campaign "$CAMPAIGN_DIR/campaign.json" \
  --out "$CAMPAIGN_DIR/structures/my-wrong-pair.pdb"
```

The command refuses when both sources resolve to one file, and when the two named binder chains carry the same sequence. Either case would rebuild the complex rather than mismatch it.

Whether the chosen binder binds the target is your assertion, not a measurement. `--campaign` buys one narrow check instead. The chosen binder is compared against every positive control the campaign declares and can resolve, and a match is refused, because the control builder would refuse it at `control-calibration`. The summary reports how many positives were compared, and the count is zero without `--campaign`. Pass `--campaign` on both subcommands.

### What each control establishes

| Kind | `role` | Establishes | Does not establish |
| --- | --- | --- | --- |
| Sequence decoy | `sequence-decoy` | Whether the predictor's interface score responds to the binder sequence, or only to the pose it was handed. | Binding specificity. Scrambled controls have scored Boltz iPTM 0.80 to 0.89 and ipSAE up to 0.65, recorded in [Tool catalogue](tool-catalogue.md). A decoy that scores high is evidence about the predictor. |
| Cross-pair mismatch | `matched-wrong-pair` | Whether the predictor reports a confident interface for a target and a binder that were never observed together. | That a real binder is selective for this target. That varies the binder, not the target. A paralog counter-screen through `scoring.counter_screen` varies the target. |
| Supplied complex | `alternate-site-complex`, `matched-wrong-pair` | Whatever the deposited complex itself establishes, which is the strongest of the three. | Nothing beyond what its own provenance supports. |

A decoy is not a substitute for a real positive control. A panel separates only when its positive is a known binder for the same target and site. Neither decoy makes a weak positive stronger.

### A negative control you already hold

A deposited non-binding or alternate-site complex needs no construction. Declare it with `structure_path`, its `target_chain`, and its `binder_chain`, and leave every source-control identifier out. The control builder validates a supplied negative at least as strictly as a positive: both declared chains must carry protein residues, the declared digest must match the file, and a negative whose binder sequence equals a positive's is refused.

`data/demo/tslp/campaign.json` shows this shape. Its negative is the deposited TSLP complex with IL-7Ralpha, whose interface overlaps the design site by a Jaccard index of 0.02.

### What this will not build

It will not build a **pose decoy**. A pose decoy moves the binder and leaves its sequence unchanged, and the control builder refuses any negative whose binder sequence equals a positive's. The rescore folds the binder from `sequence_path` and reads the pose only as the reference for the pose metrics, so a same-sequence negative would be handed the same fold as the positive. `pose-decoy` remains in `controls.supported_roles`, and a pose decoy built from a structure that is not a configured positive is the only shape that reaches the builder.

It will not choose thresholds. `--campaign` copies `scoring.thresholds.negative_control_maximum_ipsae_min` into the emitted gate. Without it the gate carries `__REQUIRED__`, which fails `check` until a control-calibration run measures a value.

It will not build a cross-pair from mmCIF. Both sources must be PDB, and the refusal names that limit. A cross-pair from an mmCIF target is unresolved.

### When the panel is wrong

`control-calibration` refuses an invalid panel before it calls any model. It is the first paid stage in the `small-run` plan, and `materialize_controls` runs before the first predictor subprocess, so a negative that reuses a positive's structure, carries a digest that does not match its file, or names a chain the structure lacks costs nothing. The refusal names the control and the field.

### Run with no panel at all

`controls.minimum_positive_controls` and `controls.minimum_negative_controls` may both be `0`, and a
campaign then validates with no control structures. Setting only the two minimums fails with
`controls.minimum_positive_controls must be at least 1`. Five fields have to agree:

- `profile.claim_level` is `"candidate"`
- `profile.scores_ungated` is `true`
- `controls.minimum_positive_controls` is `0`
- `controls.minimum_negative_controls` is `0`
- `controls.positive` and `controls.negative` hold no enabled entry

A half panel does not qualify. Leaving one enabled negative in place while both minimums read `0`
restores both refusals, because `control_panel_is_disabled` requires an empty panel on both sides.

A run that reproduces the published baseline cannot use this. `is_candidate_claim`
requires baseline fidelity to be false. Baseline fidelity requires both the
`profile.baseline_fidelity: true` declaration and matching published tool
bindings, so a declared, matching baseline roster keeps the panel required.

The run drops the `control-calibration` stage and every dependency and input that named it. Each
score row records `score_gating.mode` as `ungated` and carries the sentence `No candidate
eligibility threshold was applied to this score.` Nothing in the output supports a claim that a
design separated from a negative, because the run measured no negative.

Use it for a first pass at an unfamiliar target, for a budget probe, or when you hold no complex to
build a decoy from. The published campaign this package reproduces ran with the panel on.

## Commands

Set these paths after populating `campaign.json` and `gates/<target-id>.json`. The commands generate `qualification.json` from the composed configuration. Add explicit per-adapter qualification commands to that file before requesting a quote. The commands write to a separate roster path to preserve the packaged default roster.

The block below runs `check` partway through, and `check` reads two things you must add first. Add a `qualification.adapters` object to `$CONFIG`. Keys are the enabled adapter IDs. Each value must contain a `command_argv` list and a `receipt_path`. Each command must produce the fields that `qualify.py` validates before writing `PASS`. Add `runtime.validation_gate_dir` with the absolute path to `$CAMPAIGN_DIR/gates`. The released `small-run` profile does not declare per-adapter canary commands.

```bash
export BINDER_ROOT="$(pwd)"
export BINDER_SKILL="/path/to/installed/claude-binder-lane"
export PACKAGE_ROOT="$BINDER_SKILL/claude_binder"
export CAMPAIGN_DIR="$BINDER_ROOT/my-target-campaign"
export COMPOSED="$CAMPAIGN_DIR/composed.json"
export CONFIG="$CAMPAIGN_DIR/qualification.json"
export ROSTER="$CAMPAIGN_DIR/data/model-roster.json"
export EVIDENCE="$CAMPAIGN_DIR/data/roster-evidence"
export RUN_ID="<campaign-run-id>"
export DATA_ROOT="$CAMPAIGN_DIR/runtime"
export BUNDLE="$DATA_ROOT/runs/$RUN_ID/run-bundle"

PYTHONPATH="$BINDER_SKILL" python3 -m claude_binder.lane compose \
  --campaign "$CAMPAIGN_DIR/campaign.json" \
  --profile "$PACKAGE_ROOT/data/templates/profiles/small-run.template.json" \
  --out "$COMPOSED" \
  --json

cp "$COMPOSED" "$CONFIG"

PYTHONPATH="$BINDER_SKILL" python3 -m claude_binder.lane check \
  --config "$CONFIG" \
  --check-paths \
  --json

PYTHONPATH="$BINDER_SKILL" python3 -m claude_binder.lane qualify \
  --config "$CONFIG" \
  --output "$ROSTER" \
  --evidence-root "$EVIDENCE" \
  --canary-count 3 \
  --max-cost-usd 2.06 \
  --json

PYTHONPATH="$BINDER_SKILL" python3 -m claude_binder.lane qualify \
  --config "$CONFIG" \
  --output "$ROSTER" \
  --evidence-root "$EVIDENCE" \
  --canary-count 3 \
  --max-cost-usd 2.06 \
  --confirm-cost \
  --json

PYTHONPATH="$BINDER_SKILL" python3 -m claude_binder.lane materialize \
  --config "$CONFIG" \
  --out "$BUNDLE" \
  --data-root "$DATA_ROOT" \
  --json

PYTHONPATH="$BINDER_SKILL" python3 -m claude_binder.lane gate-scaffold \
  --config "$BUNDLE/config.resolved.json" \
  --target "<target-id>" \
  --out "$CAMPAIGN_DIR/gates/<target-id>.json" \
  --json

PYTHONPATH="$BINDER_SKILL" python3 -m claude_binder.adapters.runtime_validator run \
  --stage runtime-check \
  --phase single \
  --attempt-dir "$BUNDLE/runtime-check" \
  --receipts-dir "$BUNDLE/receipts" \
  --artifact-root "$BUNDLE/artifacts" \
  --config "$BUNDLE/config.resolved.json" \
  --plan "$BUNDLE/run-plan.json" \
  --roster "$BUNDLE/inputs/model-roster/model-roster.json"
```

The `validation_gate_dir` key resides inside the `runtime` object. Set its value to the absolute path of `$CAMPAIGN_DIR/gates`. Production scoring reads this directory. Control-calibration stages can run before a PASS gate exists. Record control measurements, write the gate file, and resume at the scoring stage.

`--max-cost-usd` is required whenever a canary would dispatch, because a canary is a paid provider job. `--confirm-cost` confirms a price you have already seen and binds no number, so on its own it let a repriced arm run at whatever it had become. A call in which every enabled arm supplies a receipt dispatches nothing and needs no ceiling.

The first `qualify` command dispatches no remote jobs. It returns the cost quote with `ok: true` and `status: awaiting-cost-confirmation`, which is a priced quote waiting on you rather than a failure. It returns `ok: false` in two cases: an arm with no recorded price, named in `errors` and in `cost.unpriced`, and a total above `--max-cost-usd`. Review the quote before running the second command. The ceiling above is a worked example for `small-run.template.json` at three canaries, where `cost_quote` returns $2.052098442. It is rounded up because the estimate is a floating-point value. Take your own ceiling from the quote the first command returns, not from this page. Adapter prices change when a route is repriced, and a stale ceiling here refuses the second command rather than dispatching it. The second command writes a fresh roster only when every canary receipt and output passes local evidence checks.

`materialize` copies the selected roster and its evidence into `inputs/model-roster/`. It records hashes for both in the run identity. Runtime validation reads the copied roster, matching `target_id` and `target_structure_sha256` against the materialized primary target.

## Cost record

The default canary count is three designs per enabled model arm. The base profile carries a per-design rate for each arm it can price, and the quote states the evidence behind every one. The three arms the `small-run` profile enables price on fal: RFdiffusion3, $0.063990 per design; ProteinMPNN, $0.004309 per sequence; ESMFold2-Fast, $0.615734 per fold. Three canaries per arm yield $0.191971, $0.012926, and $1.847201, for an estimated total of $2.052098442.

Every one of those three is a modeled estimate rather than a settled charge. Each divides a measured client wall time for one unit by the reference per-second rate for the machine that ran it, and `qualification.cost_basis` on the adapter records the seconds, the machine, the rate, and where the measurement came from. Client wall time brackets runner uptime from above, so each rate quotes high rather than low. No provider invoice line has been read against these three runs, so treat the total as a ceiling to plan under and not as a price you have been charged.

When an arm carries no rate, `qualify` names it in `cost.unpriced` with the reason, reports a priced subtotal for the rest, and leaves `total_usd` null, because a total that omits an arm is not a total. All three arms this profile enables carry a rate, so the quote returns a total and `cost.unpriced` is empty.

Do not transplant the `$/design` figures in `$PACKAGE_ROOT/data/model-roster.json` onto another provider. They rest on `reference-gpu-rates-2026-08-23.json`, which states the rule in its own `source` field: those rates were read from one provider usage export on 2026-08-23, the machine names are that provider's, and an unknown price is safer than a transplanted one. Measure your own rate and supply it through `--cost-rate`, which outranks the shipped rate and is labelled `operator override` in the quote.

A fresh provider invoice remains unproven until the provider settles billing for the canary run.

## No literature binder

When no literature binder exists, record the search as `literature_control_search` with `NO_CONTROL_FOUND`. Set that arm and the gate to `FAIL`. The gate preserves the search record and refuses production scoring because condition (b) has no positive-control measurement. A target without a suitable positive control cannot receive a PASS production scoring gate under this protocol.

## Paid-call boundary

The local test suite verifies command rendering, target hashing, roster writing, roster bundling, and runtime validation offline. Verifying provider execution, target-fold recapitulation, positive-control separation, and billing settlement requires paid provider calls.
