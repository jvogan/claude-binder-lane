# The published workflow, and where this package sits in it

The release describes an operating structure, not just a tool roster. This page condenses that structure and maps each step onto a command here. Read it to see what a campaign does end to end, and which parts this package automates.

Source: the [released prompt](https://huggingface.co/datasets/Anthropic/claude-protein-binder-design/blob/d442eeb/prompts/prompts/multi_target_binder_design_prompt.md) at `d442eeb`, CC BY 4.0, read 2026-09-16.

## Per target, start to finish

```
 PREPARE THE TARGET
   1  Dossier             Oligomeric state the release evaluated, cofactors and
                          ligands, the exact construct, and which
                          depositions show the epitope ordered.
   2  Construct and site  One construct per target, matched to that system.
                          Epitope from interfaces already explored in the
                          design literature.

 BUILD THE INSTRUMENT
   3  Tool bring-up       Install, pin, and run an N=1 canary that parses.
   4  MSA staging         One a3m per construct, cached and read by every arm.
   5  Validation gate     (a) The co-folder reproduces the target structure.
                          (b) A known binder scores above every negative.

 ======  Steps 8 to 10 are blocked until step 5 writes PASS to disk.  ======

 PRODUCE
   6  Generation wave     At least 50 backbones per starred method per target.
   7  Pre-score filter    Novelty, liabilities, foldability, and redundancy.
                          Runs before any co-folding spend.
   8  Screen, 1 seed      Rank by the 4:1 ipSAE to sc_DockQ weighted z-score.
   9  Promote             Higher seed tiers on the survivors. Rank on ipSAE and
                          sc_DockQ together, never ipSAE alone.
  10  Final rank          rank_zscore over six terms: three ipSAE_min weighted
                          4, three sc_DockQ weighted 1.
  11  Receipts            Design-count ledger and spend ledger. One writer per
                          subfile, writes idempotent on (job_id, stage).
```

Each step maps to a page or a command here.

| Step | Where to read it | What to run |
| --- | --- | --- |
| 1 Dossier | [Published targets](published-targets.md), [Target inputs](target-inputs.md) | |
| 2 Construct and site | [Published targets](published-targets.md) | `make_target_inputs` |
| 3 Tool bring-up | [Tool bring-up](tool-bringup.md) | `lane contract-dry-run`, `generator_preflight` |
| 4 MSA staging | [Platform tool discovery](platform-tool-discovery.md) | |
| 5 Validation gate | [Target qualification](target-qualification.md), controls in [Published targets](published-targets.md) | `lane gate-scaffold`, `make_negative_control` |
| 6 Generation wave | [Running a campaign](running-a-campaign.md) | `lane compose`, `lane materialize`, `lane execute` |
| 7 Pre-score filter | [Design and filters](design-and-filters.md) | |
| 8 Screen, 1 seed | [Nine decisions](the-nine-decisions.md#8-rounds-and-stopping) | `lane rank`, `lane sequence-score-table` |
| 9 Promote | [Nine decisions](the-nine-decisions.md#8-rounds-and-stopping) | |
| 10 Final rank | [Scoring details](scoring-details.md) | `lane rank` |
| 11 Receipts | [Approval and spend](approval-and-spend.md), [Reading results](reading-results-in-claude-science.md) | `lane reconcile-spend` |

`lane execute` writes one receipt per stage to `artifacts/receipts/`, and the `render-viewer` stage writes the offline viewer. There is no single report command.


**No threshold ships, and none could.** The release publishes the gate requirement and the control molecule for each target. It publishes no threshold file, and its own tables carry none: the in-silico tables are co-fold predictions, epitope contacts, provenance and constructs. That is not an omission. A gate threshold is frozen per target per instrument at gate time, and the release states that its z-scores are transductive and comparable only within the pool used to compute them. A number carried off one target, or off one arm, measures nothing on another.

**The one gate record in this package is an example of shape, not a threshold to use.** `claude_binder/data/gates/pdl1.json` carries `status: UNVERIFIABLE` and says why: the historical panel measurements ship without the 4ZQK-derived structure and residue-map artifacts needed to re-derive them. The gate machinery never loads it. `lane._validation_gate_directory` resolves a directory only from a caller-supplied `validation_gate_dir`, `gate_dir`, or `state_root`, and returns nothing when none is given, so no path reaches the packaged file. It blocks nothing and gates nothing. Read it for the fields a gate record carries, and build your own target's panel with `make_negative_control` and your own controls.

Quoting a number out of that file, or out of the historical calibration prose in [measured costs](measured-costs.md), as though it were a threshold for another target is the live failure mode here. Those six values ship in no file and no code reads them.

**The gate is the load-bearing step.** Production scoring is blocked until a per-target gate file says PASS. A verbal claim does not satisfy it. That is why a campaign runs its control before it runs its designs, and why a control failure is cheap news rather than a wasted round.

## The instrument

Three co-folding arms, one sample per seed, template-free, target-chain MSA, binder single sequence: ESMFold2-Full, ESMFold2-Fast, Protenix v2. Ranking turns on the count of arms rather than their names, so a substitution that keeps three arms keeps the ranking form. When an arm is unavailable or unvalidated on a target, the release substitutes one independent-lineage co-folder per missing arm, preferring AlphaFold-Multimer-v3, then AlphaFold3 code with OpenFold3 weights, then Chai-1, then Boltz-2 or Boltz-1.

## What the release runs that this package does not

The release is a multi-agent campaign. It carries roles this package has no equivalent for, and a single-operator campaign does not need most of them.

| Release role | What it owns | Here |
| --- | --- | --- |
| Orchestrator | Every obligation in the prompt | The Claude Science session |
| DISPATCH sub-agent | The only issuer of submit tokens; recomputes the (target, method) floor matrix every 15-minute cycle | No equivalent. `lane execute` dispatches directly |
| BUDGET sub-agent | All metered-spend reads, a standing 15-minute cycle, a scale-up trigger when spend lags elapsed time by 20 points | Partly. `qualify` and `budget_plan` price forward and invert a ceiling; nothing runs on a cycle |
| DELIVERABLES coordinator | The single writer to shared storage | No equivalent |

The parts worth keeping at small scale are the single-writer ledger rule, the floor matrix as an obligation tracker, and the pacing rule that generation stays level with scoring because an unscored design is wasted compute.

## Scale, and what a small campaign changes

The release covers 14 targets, 50 backbones per starred method per target, 30 designs selected per target, and 100,000 to 1,000,000 designs generated and screened campaign-wide. A campaign at a few dozen designs reproduces the method and the shape, not the protocol. That is the second campaign shape in [published targets](published-targets.md), and it makes no baseline fidelity claim, which stops no stage.

Thresholds do not carry across that gap. They are frozen per target at the validation gate, so a published number is a record of one target's controls rather than a setting to copy.
