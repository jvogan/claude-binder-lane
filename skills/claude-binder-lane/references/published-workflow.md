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
   4  MSA staging         Cache an a3m per construct for MSA-capable arms.
                          ESMFold2-Fast runs single-sequence.
   5  Validation gate     (a) The co-folder reproduces the target structure.
                          (b) A known binder scores above every negative.

 ======  Production design scoring starts after step 5 writes PASS.  ======

 PRODUCE
   6  Generation wave     At least 50 backbones per starred method per target.
   7  Pre-score filter    Novelty, liabilities, foldability, and redundancy.
                          Runs before candidate co-folding spend.
   8  Screen, 1 seed      Rank by the 4:1 ipSAE to sc_DockQ weighted z-score.
   9  Advance survivors   Send a budgeted fraction of screened designs to the
                          deeper tier.
  10  Intermediate        Five seeds per predictor on survivors; choose parents
                          with ipSAE and paired sc_DockQ together.
  11  Optimize            Score each new round with the parent pool's predictor
                          set and seed count; choose the next parents.
  12  Final rank          rank_zscore over six terms: three ipSAE_min weighted
                          4, three sc_DockQ weighted 1.
  13  Receipts            Design-count ledger and spend ledger. One writer per
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
| 8 Screen, 1 seed | [Scoring details](scoring-details.md#seed-tiers) | `cofold-screen-*`, `score-screen` stages |
| 9 Advance survivors | [Nine decisions](the-nine-decisions.md#8-rounds-and-stopping) | `screen-survivors` stage; set `cofold.intermediate_fraction` |
| 10 Intermediate | [Scoring details](scoring-details.md#seed-tiers) | `cofold-intermediate-*`, `score-intermediate`, `promote` stages |
| 11 Optimize | [Running a campaign](running-a-campaign.md) | `optimization-plan-round-*`, `optimization-cofold-round-*`, `optimization-measure-round-*`, `optimization-select-round-*` stages |
| 12 Final rank | [Scoring details](scoring-details.md) | `cofold-rescore-*`, `uniform-rescore`, `final-rank` stages |
| 13 Receipts | [Approval and spend](approval-and-spend.md), [Reading results](reading-results-in-claude-science.md) | `lane reconcile-spend` |

These stage IDs belong to the full campaign profile. `lane execute` records
each completed stage in `artifacts/receipts/`; `render-viewer` writes the
offline viewer. The BindCraft2 small campaign uses its separate
[fast path](small-campaign-fast-path.md), which scores its declared roster
without a screen-survivor tier.


## Target-specific validation

The released prompt gives a default pose-DockQ threshold of 0.23 and allows a
stricter threshold to be frozen at the target's validation gate. Establish the
control-separation rule on the chosen target and predictor instrument. Scores
normalized within one design pool cannot serve as control thresholds for a
different pool.

The full campaign requires a per-target gate file with `status: PASS` before
production scoring. The packaged `claude_binder/data/gates/pdl1.json` file
shows the record format and has `status: UNVERIFIABLE`; it supplies no gate
threshold for a new campaign. Run the target's positive and negative controls
and save their gate record before scoring designs.

## The instrument

The published instrument uses three co-folding arms, one sample per seed, and
no template injection. ESMFold2-Fast runs single-sequence. ESMFold2-Full and
Protenix v2 use a cached target-chain MSA; the binder is single-sequence.
The published score needs three modes across at least two model lineages.
A substituted arm keeps the ranking form when that count and lineage requirement
hold. When an arm is unavailable or
unvalidated on a target, the release substitutes one independent-lineage
co-folder per missing arm, preferring AlphaFold-Multimer-v3, then AlphaFold3
code with OpenFold3 weights, then Chai-1, then Boltz-2 or Boltz-1.

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

The released multi-target prompt lists 14 targets, requires 50 scored backbones
per starred method per target, requests 30 designs per target, and calls for
100,000 to 1,000,000 designs generated and screened campaign-wide. Anthropic's
[study post](https://www.anthropic.com/research/Claude-accelerates-protein-design)
reports 15 targets attempted and binders found for 14. A campaign at a few
dozen designs follows the method at smaller scale; record that scale in its
claim. [Published targets](published-targets.md) distinguishes the target lists.

Thresholds do not carry across that gap. They are frozen per target at the validation gate, so a published number is a record of one target's controls rather than a setting to copy.
