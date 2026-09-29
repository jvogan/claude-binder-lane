# Open questions

Read this at kickoff. Protocol disclosure rules require a recorded decision on every entry before scoring begins.

Twelve items require decisions. Six items lack published answers in the protocol or release notes. Six items are inline parameter choices, and the table below names the file or configuration key that sets each one.

## Contents

- [Six questions with no published answer](#six-questions-with-no-published-answer)
- [How to cite `ipSAE_min`](#how-to-cite-ipsae_min)
- [The six inline decisions](#the-six-inline-decisions)
- [Pre-registering your choices](#pre-registering-your-choices)
- [What packaging still leaves open](#what-packaging-still-leaves-open)

## Six questions with no published answer

Four questions affect score computation: the z-score estimator, normalization cohort membership, the ipSAE PAE cutoff, and target-specific ipSAE thresholds. The first two apply only when a profile uses the published three-mode z-score; a disclosed candidate-level raw mean has no normalization pool. Shared volume write visibility affects wave barrier synchronization. Replication tolerance determines whether distributions match published campaign baselines.

Specify the population (n) versus sample (n-1) standard deviation denominator for transductive z-scores. In the meantime, the package divides by n in `ranking_normalization`, whose docstring calls its output candidate-only population parameters (`claude_binder/lane.py`). Every published three-mode rank score reads that scale; a candidate-level raw mean does not. Disclose the applicable choice in campaign metadata before production scoring.

Clarify normalization cohort definition in published protocol specifications. In the meantime, the package restricts normalization to candidates that are coverage-complete and filter-passing (`claude_binder/lane.py`), and passes only those rows to `ranking_normalization` (`claude_binder/lane.py`).

Document the exact ipSAE PAE cutoff used during initial campaign generation. In the meantime, use the 10 angstrom cutoff with `d0res` from the released ten-predictor re-scoring, which the package carries as the `compute_ipsae` default `IPSAE_INTERFACE_CUTOFF_ANGSTROM = 10.0` (`claude_binder/adapters/binder_metrics.py`) and implements as Dunbrack 2025 Equation 14 in the d0res variant (`claude_binder/adapters/binder_metrics.py`).

Verify shared volume write-visibility semantics across sequential provider container waves. In the meantime, route ledger writes through singletons with fresh volume mounts to avoid stale mount snapshots.

Establish target-specific calibrated ipSAE thresholds. No published threshold exists for ipSAE on either the maximum or the minimum. A threshold of 0.30 circulates in a 2026 screening paper that attributes it to the 2025 preprint. That threshold does not appear in the preprint, and the paper set it for the maximum ipSAE instead of the minimum. It is not a usable default. In the meantime, derive empirical cutoffs from positive- and negative-control separation at the validation gate.

Establish explicit statistical tolerance bands for replication claims. In the meantime, compare empirical medians directly against per-target values in `references/released-record.md`.

## How to cite `ipSAE_min`

This note covers attribution and records no decision. The `ipSAE_min` metric is defined in the protocol. The protocol attributes the minimum to the 2025 preprint, but that preprint defines chain-pair ipSAE as the **maximum** across both directions. Its reference implementation reports asymmetric and maximum rows without a minimum. The minimum represents a more conservative variant. Cite the protocol for the minimum variant and the preprint for the base metric.

## The six inline decisions

The protocol requires explicit selection and freezing of these six values:

| What you have to choose | Where it is set |
|---|---|
| What `out` and `novelty` hold, and which volumes a job mounts | [Modal bring-up](modal-bring-up.md), the Volume mapping a submission carries |
| Whether the volume split your route needs is required | [Modal bring-up](modal-bring-up.md), the Volume mapping a submission carries |
| The CA-RMSD threshold for target fold recapitulation | [Campaign fields](campaign-fields.md), `scoring.implementations.maximum_target_alignment_rmsd` |
| The numeric margin for positive-control separation | `qualification.control_separation.thresholds`, keyed `<arm>:<target_id>`, in your `campaign.json` |
| The structural-plausibility thresholds | [Design and filters](design-and-filters.md), set under `filters.contracts` in `campaign.json` |
| Per-design wall clock and cost, per model in your funnel | [Compute and pacing](compute-and-pacing.md) |

This package supplies no separation margin. `DEFAULT_THRESHOLD` in
`claude_binder/control_separation.py` is `None`, and the comment beside it records why:
calibration covers one target and fifteen control observations. With no threshold set,
`configured_threshold` returns that unset default with the source text `no default threshold is
configured`, and a stored measurement then passes on presence alone, because `assess_roster` requires
the value to reach a threshold only when one exists. Set one under
`qualification.control_separation.thresholds` keyed `<arm>:<target_id>`, or under
`qualification.control_separation.threshold` to cover every pair.

TODO(evidence): no file in this package states a numeric positive-control separation margin for
any target. Settled by: a validation-gate run on your own target that folds positive and negative
controls on the same arm, construct, and seed count, then records the resulting `hedges_g` value
under `qualification.control_separation` and the threshold that value supports.
[Stored control separation](campaign-fields.md#stored-control-separation) documents every key in
that block and what each command does with it.

`optimization.early_stop_margin` does not answer this question, and neither does
[scoring details](scoring-details.md). The early-stop margin is a stopping distance on the
campaign's objective metric, and [the nine decisions](the-nine-decisions.md) records that a margin
belongs to one metric, so a campaign may not carry it to another. Scoring details never uses the
word margin; it leaves open the pair of positive-control contact thresholds recorded under
[What packaging still leaves open](#what-packaging-still-leaves-open).

## Pre-registering your choices

Record decisions for all twelve items in the gate file or campaign configuration before starting production scoring. Announce these parameters in the kickoff record. The z-score estimator and normalization cohort alter all sheet values, requiring pre-run registration.

## What packaging still leaves open

These configuration and packaging items are conditional prerequisites. Resolve an
item before selecting the named tool or route, or before making the associated
claim; an initial execution that does not use that tool or route is not blocked
by an unrelated item.

| Scope | What is missing | Where it is recorded |
|---|---|---|
| Only when `protenix-v2` is selected | Access to the official Protenix v2 checkpoint and its captured SHA-256 manifest | `references/protenix-v2-bringup.md`, and again in `envs/binder_scoring_gpu.py` |
| Only when `protenix-v2` is selected | The completed Protenix v2 hash lock, image recipe, and offline GPU qualification | `references/protenix-v2-bringup.md`, and again in `envs/binder_scoring_gpu.py` |
| Only when the GPU scoring image uses `ipsae.py` | The commit to pin `ipsae.py` at | `envs/binder_scoring_gpu.py` |
| Only when `protenix-v2` is selected | Measured Protenix v2 artifact hydration and job-time no-egress validation | `references/protenix-v2-bringup.md` |
| Only when novelty filtering uses the local MMseqs2/UniRef90 route | MMseqs2/UniRef90 full-database memory, disk, wall-time, and search canary receipts | `references/mmseqs2-uniref90-local-route.md` |
| Only for a paid provider route that uses concurrent waves | Who raises the provider concurrency cap, and to what | `references/compute-and-pacing.md` |
| Before a run that generates designs and uses a saturation or ceiling claim | When the ceiling calibration band becomes required | `references/scoring-details.md` |
| Before a target-specific positive-control contact gate is used | The two positive-control contact thresholds | `references/scoring-details.md` |
| Before `sc_DockQ` carries ranking weight over backbone-only design poses | The `fnat` contact cutoff for a reference pose with no side chains, and the smallest native contact count a comparable score may rest on | [The backbone-only pose metric](#the-backbone-only-pose-metric) |

## The backbone-only pose metric

Neither setting below is rank-bearing on any route that has run. A cutoff sweep
over four harvested folds on 2026-09-07 held the rank order at every cutoff from
5 to 12 Angstrom, and native contact counts at the packaged 5.0 Angstrom cutoff
were 22, 21, 29 and 13. No backbone-only design pose exists anywhere in the
package. All 1047 parseable structures were counted, design poses carry 7.62 to
8.01 heavy atoms per residue, and the six structures below 5.5 heavy atoms per
residue are test fixtures of 2 to 8 residues. The case these two settings govern
has never been scored.

The mechanism they guard is still open. `compute_dockq` divides matched contacts
by the native contact count and refuses only when that count is zero. One native
contact pair therefore yields an `fnat` of exactly 0.0 or 1.0, and `fnat` is a
third of DockQ. `compute_dockq` returns no contact count, so a thin denominator
is invisible to every consumer of the score.

Two settings decide this, and both are the scientist's:

1. The `fnat` cutoff for a reference pose carrying no side chains. The function
   takes `fnat_cutoff_angstrom`, no caller passes another value, and no profile
   field reaches it. A backbone-appropriate cutoff is wider than 5.0 Angstrom, and
   this package records no measured value for it.
2. The smallest native contact count at which an `sc_DockQ` may be compared
   between designs. Below it the score should be recorded and excluded from the
   ranking rather than treated as a measurement.

Packing side chains onto the design pose is the third option and is unavailable
here. No side-chain packer ships in this package.

Decide both before `sc_DockQ` carries weight in a multi-term ranking over
backbone-only poses. Both questions become live the first time the RFdiffusion or
RFdiffusion3 generator arm harvests backbones, and not before.

### The 0 to 8 contact range is superseded

An earlier version of this entry reported native contact counts running from 0 to
8 across five designs, and an `sc_DockQ` of 0.91 resting on one native contact
pair. Neither figure reproduces from harvested evidence. The 0.91 is
`design-014`'s `sc_dockq` of 0.9172 in a 2026-08-31 fal cohort report, whose two
neighbouring columns hold a clash count of 1 and a site contact IoU of 0.6667.
Neither of those columns is a contact count. The same candidate folded on Modal
scores 0.970 on 21 native contact pairs.
