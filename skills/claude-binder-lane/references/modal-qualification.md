# Qualify a Modal arm

A Modal arm is qualified from a receipt a Claude Science frame harvests, not from a canary the
qualifier dispatches. `lane qualify --receipt <adapter>=<path>` turns that receipt into a roster row.
Every other provider still uses the ordinary dispatch path in
[Qualify a target-specific model roster](target-qualification.md).

## Why the ordinary path cannot run a Modal canary

`claude_binder.qualify._run_adapter` runs each canary command through `subprocess` with
`shell=False`. Three facts make that subprocess unable to reach Modal.

- The adapter programs carry no Modal route. `esmfold2_predictor.py` and
  `esmfold2_fast_predictor.py` name neither `CLAUDE_BINDER_EXECUTION_ROUTE` nor `modal_platform`.
  They are in-container programs. The route is applied by the stage executor in `lane.py`.
- Modal submission needs the `host` object. `modal_platform.submit_stage` calls
  `host.compute.create("modal", ...)`, and `host` exists only inside the Claude Science kernel. A
  subprocess of that kernel does not hold it.
- Modal reports completion out of cell. `dispatch_modal.submit_wave` returns without waiting, and
  `job.result()` raises `JobPending` until the notification arrives. No synchronous executor can
  wait for one.

The runtime validator refuses an enabled Modal arm whose roster row is not `PASS` for the campaign's
own target. Without a way to write that row, a Modal campaign could not pass its own
`runtime-check`, so no Modal stage could ever be reached.

## What the canary must measure

The canary folds 2 to 4 designs. `runtime_validator._run_check` refuses any other count, so a single
fold cannot qualify an arm.

The receipt is one JSON object. `build_row` fills the campaign-derived and timing fields itself. The
canary supplies the rest.

| Field | Value |
| --- | --- |
| `exit_code` | integer, and `0` for a `PASS` row |
| `n_designs` | 2 to 4 |
| `output_paths` | the structure files, relative to the evidence root or the canary run directory |
| `output_shape` | what those files contain |
| `output_residue_count` | required for mmCIF, an integer of at least 30, counted from the written file |
| `image_id` | the image the job ran, matching `resources.container_image_digest`, added by the submitter |
| `gpu_type` | the device read inside the container |
| `weights_sha256` | the checkpoint digest, or `not_applicable: <reason>` |
| `weights_revision` | required when `weights_sha256` records an absence |
| `container_build` | `true`, with the probe output in a sibling evidence field |
| `package_imports` | `true`, with the probe output in a sibling evidence field |
| `flags_match_production` | `true`, with the compared configuration in a sibling evidence field |
| `sequence_adapter_consumed` | `true` |
| `sequence_output_count` | the count of sequences folded |
| `wall_clock_s` | measured seconds, because no process here watched the run |

Two fields the container cannot supply on its own.

`image_id` is a submission-side fact. A container is launched onto an image and is never told which
one, so it cannot measure this. The submitter adds it after the harvest, from the provider params of
the job that produced the outputs, and records the job id beside it so a reader can trace the pair.

`output_residue_count` decides the structure check for an mmCIF output. The check reads the atom_site
chains out of the file itself, then requires the row to record a residue count of at least 30, and an
absent count fails every file. Count it out of the mmCIF the fold wrote, not out of the input
sequence lengths.

`weights_revision` is the field to get right. `runtime_validator._revision_pin_check` reads every
`repository@revision` pair out of it and requires that set to equal the set in the profile's
`model_revision`. The packaged fal roster records two pins where the ESMFold2-Fast profile pins one,
which is why that row fails the check. Read the revision the container resolved and report exactly
the repositories the profile names.

A run that records `weights_checkpoint_revision` as a TODO did not read the revision inside the
container, and it cannot qualify an arm.

## What one canary measured

One job ran this procedure on 2026-08-31 and produced a `PASS` Modal roster row. It folded three
designs against a 115-residue target on an A100-80GB, exited 0, and took 72
job seconds of which the model load was 24.8. The container reported torch 2.7.1+cu126, transformers
4.57.6, python 3.12.1, and device `NVIDIA A100-SXM4-80GB`. It resolved `biohub/ESMFold2-Fast` at the
commit the profile pins, read through `config._commit_hash`. `huggingface_hub` could not be reached
because the container sets `HF_HUB_OFFLINE`, so `config._commit_hash` is the read that works there.
No cost surface returned a figure for the job.

Record the job id in the run's own evidence directory, never on a page you share. A provider job id
identifies the account.

## The frame procedure

The frame owns `host`, so it runs the canary and harvests its receipt.

1. Submit the canary with `dispatch_modal.submit_wave`, end the cell, and park on the notification.
2. Call `dispatch_modal.close_wave` and confirm its barrier.
3. Write the harvested structures under `<evidence-root>/canary/<adapter-id>/`.
4. Write the receipt beside them, covering every field in the preceding table.

Then write the roster from a shell. Define the four paths first. `BINDER_SKILL` is the
installed skill directory, the one that holds `kernel.py`. `BUNDLE` is the materialized
run bundle. `RUN_ROOT` is the run directory the roster is written into. `EVIDENCE_ROOT`
is where step 3 above wrote the harvested canary structures.

```sh
export BINDER_SKILL="/path/to/installed/claude-binder-lane"
export BUNDLE="/path/to/run-bundle"
export RUN_ROOT="/path/to/run-root"
export EVIDENCE_ROOT="/path/to/roster-evidence"
```

```sh
PYTHONPATH="$BINDER_SKILL" python3 -m claude_binder.lane qualify \
  --config "$BUNDLE/config.resolved.json" \
  --output "$RUN_ROOT/model-roster.json" \
  --evidence-root "$EVIDENCE_ROOT" \
  --canary-count 3 \
  --receipt esmfold2-fast-predictor="$EVIDENCE_ROOT/canary/esmfold2-fast-predictor/receipt.json"
```

No cost confirmation and no `--max-cost-usd` are required when every enabled arm supplies a
receipt, because this call dispatches nothing.

**Supply a receipt for every enabled arm, not just one.** An arm left without one does not fall
back to the ordinary cost gate here, because no shipped Modal profile declares a qualification
`command_argv`, so there is nothing for that gate to dispatch. Leaving one out refuses twice
over: first for the missing ceiling, quoting a total the run will never spend, and then with
`enabled model arms have no explicit qualification command_argv: <the arms you left out>`.

The command above supplies one receipt, so it completes only on
`supplied-candidates-modal.template.json`, which enables one qualification arm. The three-arm
Modal profiles need one `--receipt` each for `rfdiffusion-generator`, `proteinmpnn-designer` and
`esmfold2-fast-predictor`, harvested by the frame procedure above, one arm at a time.

The written roster records which rows came from where:

```json
"dispatch": {"dispatched_here": [], "supplied_receipts": ["esmfold2-fast-predictor"]}
```

Each supplied row also carries `qualification_dispatch`, `qualification_receipt_source`, and
`qualification_receipt_sha256`.

## What a supplied receipt does not skip

Handing in a receipt skips the dispatch. It skips none of the evidence.

- `build_row` rehashes every file the receipt names and overwrites `output_sha256` with what it
  measured.
- `build_row` sets `target_id` and `target_structure_sha256` from the campaign, and raises when the
  receipt names a different target or a different structure hash.
- `write_roster` refuses to write a `PASS` row that is missing any required field.
- `runtime-check` parses the structure files itself and runs its geometry checks on them.

A receipt whose structure files do not parse fails that last check and the arm stays unqualified.
