# Route status

## Accelerated design and sequence kits, 2026-09-30

Account-local recipes and handoff guides ship for Genie3, PXDesign,
RFdiffusion3, BoltzGen, Complexa, and full-backbone ProteinMPNN. The
[dated qualification](accelerated-kit-qualification-2026-09-30.md) separates
actual native Modal inference from recipe availability, Claude Science
execution, full graph bindings, and scientific validation.

## BindCraft2 on Modal, 2026-09-21

The packaged image ran BindCraft2 on a Modal A100-80GB. The adapter toolcheck
succeeded, upstream execution produced an accepted output, and the package
parser read that output. A repeat stored its artifacts on persistent storage.

| Phase | Recorded wall time |
| --- | --- |
| Image and weights setup on CPU | 100.6 seconds |
| Adapter toolcheck on A100-80GB | 17.3 seconds |
| Upstream execution on A100-80GB | 619.0 seconds |
| Repeat with warm compilation cache | 370.2 seconds |

These observations establish that the installed tool and parser worked on that
route. Complete execution through the Binder profile and its downstream screen
remain unverified. Settled dollar cost is pending. Use the
[packaged image and setup guidance](tool-catalogue-details.md#bindcraft2-bound-on-2026-09-21)
when selecting a route. The raw contributor run artifacts are not bundled here.

## fal billing observations, 2026-09-19

Account balance changes recorded two settled charges: 0.0081026264 USD for an
M-tier toolcheck plus one sequence-design call, and 0.0843073825 USD for an
H100 toolcheck. These are separate observations from the historical ESMFold2-Fast
requests. The latter requests retain their modeled cost labels. Read
[fal cost scope](measured-costs.md#fal-esmfold2-fast-the-per-request-model-load-is-the-cost)
before using either record to estimate a new workload. Raw account billing
records are not bundled here.


## Three fal arms qualified on a deposited PD-L1 structure, 2026-09-16

Generation, sequence design and folding have all executed on a paid route against a target built
from a deposited structure. The roster records `PASS` for `rfdiffusion-generator`,
`proteinmpnn-designer` and `esmfold2-fast-predictor`, each bound to target `pdl1-4zqk`.

**The target came from 4ZQK, not from a packaged fixture.** The coordinate file was fetched from
RCSB, PD-L1 chain A residues 18 through 132 were taken as a 115-residue chain and relabelled to B,
and the PD-1 chain was relabelled to A. All three roster rows carry the resulting
`campaign_target_structure_sha256`
`0529940e33c38aa202ff606649e92c6e2ea6dd10053fb6352d035fa3f362c05d`. The package still ships no
4ZQK coordinate file, so this target was materialized outside the package by the operator, which
is the path [Target inputs](target-inputs.md) describes.

**The epitope came off the real partner interface.** `make_target_inputs --partner-chain A` ran in
`reference-partner-contacts` mode at a 5.0 angstrom heavy-atom cutoff and returned 22 surface
residues. All 17 hotspots in the published manifest appear among them.

| Arm | Units | Client wall seconds | Modeled cost |
| --- | ---: | ---: | ---: |
| `rfdiffusion-generator` | 2 designs | 337.3 | 0.127981 |
| `proteinmpnn-designer` | 2 designs | 125.0 | 0.008617 |
| `esmfold2-fast-predictor` | 2 folds | 1476.6 | 1.231468 |

**The generator-to-designer handoff is traceable by digest.** RFdiffusion3 returned two two-chain
backbones, chain A an 88-residue binder inside the 60 through 90 contig and chain B the
115-residue target. ProteinMPNN records the generator's first output digest as its
`input_structure_sha256`, designed chain A only with `fixed_chains=['B']`, and returned 87-residue
sequences at 0.6023 and 0.5227 sequence recovery.

**Budget a cold ESMFold2-Fast worker above the 1700-second bound the adapter declares.** The first
dispatch returned HTTP 500 with `WorkerFailedError` and `python exceeded 1700 seconds` at 1711.4
elapsed. The retry against the hydrated environment folded both structures, the first in 1015.3 s
and the second in 428.99 s. Model load dominates a cold call and 1700 s leaves no margin for it.
The failed call bought about 2.14 USD of provider time and returned nothing.

**The first dispatch of this run sent the target as mmCIF and the deployment answered `HTTP 500`.**
The identical coordinates written as PDB produced backbones on the next attempt. See
[Target inputs](target-inputs.md#write-the-dispatched-structure-as-pdb) for what the two files held
and why a full lane run does not meet this.

**Run `canary_runner finalize` after the arms complete.** A backbone generator finishes before
anyone can answer whether a sequence adapter consumed its output, so `rfdiffusion-generator` sits
`PENDING` on `sequence_adapter_consumed=true` and `sequence_output_count` until that step reads
which backbones the designer actually used and completes the receipt. Pass `--session` an output
path, because the flag reads like an input and is written to. It refuses a path that already holds
anything other than a session summary, so naming your composed config there is an error and not a
loss. Pass `--replace` to overwrite a file it refuses.

Every figure above is a modeled estimate at the package reference rate, not an invoice line.

## Three fal arms qualified on a packaged proxy structure, 2026-09-13

Generation and sequence design have executed on a paid route. Three adapter canaries ran on fal and
the final roster records `PASS` for all three: `rfdiffusion-generator`, `proteinmpnn-designer`, and
`esmfold2-fast-predictor`, each bound to target `pdl1-primary`.

**The structure behind those rows is a packaged coordinate proxy, not a deposited PD-L1 target.**
All three rows carry `target_structure_sha256`
`7f87db2b085795b3ea87ee4c721a2f3e33ac69948175c4eae255dd61b5822c2f`, which is
`data/roster-evidence/canary/rf3/structures/design-spec_pdl1_binder_0_model_0.cif`, a 170,603-byte
file this package ships. That file is a design-spec model of a binder-target complex. A `PASS` here
establishes that the adapter executes on fal and returns output the next stage parses. The
2026-09-16 run above supersedes it on target provenance.

**The generator-to-designer handoff is traceable by digest.** RFdiffusion3 wrote backbone
`66a533fdd282e66a8beadfcf9b5a2ed03c80c22e064344215f511bdbee311053`, and ProteinMPNN records that
same digest as its `input_structure_sha256`. The two arms passed a real artifact between them.

| Arm | Units | Client wall seconds | Modeled cost |
| --- | ---: | ---: | ---: |
| `rfdiffusion-generator` | 3 designs | 126.8 | 0.158528 |
| `proteinmpnn-designer` | 3 designs | 156.9 | 0.010986 |
| `esmfold2-fast-predictor` | 3 folds | 2083.1 | 2.603922 |

A second run the same day closed the two rows the first left `PENDING` and confirmed that both arms
send production's declared literals. ProteinMPNN compared `--sampling-temp=0.1` and `--seed=37`, and
the provider's own FASTA header records `seed=37`. ESMFold2-Fast compared `--max-seconds=1700` and
`--timeout-seconds=1850`, read off the running client argv while the fold was in flight.

Every figure above is a modeled estimate at the package reference rate, not an invoice line. No fal
invoice has been read against any of it.

Detailed receipts are not included in this distribution. These tables summarize the recorded measurements.

## What each route can run, 2026-09-07

The fal route has completed every stage of a paid campaign. Campaign `pdl1-fal-supplied-n1-v2`
committed 16 of 16 stages with `status.json` reporting `state: completed` and `ok` true. It made 16
paid calls, each carrying its own `predict-intent` and `predict-outcome` row across three journals,
split 1 screen, 10 control and 5 rescore. Stage wall times were 382.8 s for
`cofold-screen-esmfold2-fast`, 1993.0 s for `control-calibration` and 888.3 s for
`cofold-rescore-esmfold2-fast`.

No fal figure in that record is a measured cost. The run booked 14.32675 USD against a declared
15.00 USD cap, and every one of those dollars is a pre-run upper bound. the package's billing probes returned
404 during that run, so `spend.jsonl` carries `charge-estimate` rows sourced from the approval
ledger and `cumulative_settled_amount` stays 0.0.

The run produced a ranking and refused to claim it. `ranked-candidates.json` scores
`design-001` and then reports `scoring_arm_status: unvalidated`, `ranking_claim_status:
unvalidated`, and zero selected candidates, because the profile sets `production_scoring` false.

Per-call fal runtimes now exist. An earlier campaign the same day, `pdl1-fal-supplied-n1`,
recorded `total_seconds` for all ten control folds: 159.889 to 272.224, mean 191.400. Its booking
used `control-builder`'s `timing_basis` upper bound of 337.1 seconds per fold, which is 1.76 times
the measured mean. The bound held, so the booking was conservative. Both numbers
are modeled at the package reference rate of 0.00125 USD per second.

Four limits remain. The contract dry run covers one route rather than the shipped profile
surface. A run handoff instructs a fresh Claude Science frame to continue a run only its origin
frame can write. No shipped Modal profile declares a qualification canary command, while ten
profiles declare one on their fal adapters that runs once you supply your own deployed fal URL.
The estimator resolves a stage's provider for timing and drops it for rate.

The 2026-09-07 free-graph run is the current record. It reports 43 of 43 stages, 43 final
receipts, 24 PNG files and a 12-entry picture archive, with zero provider-facing calls. The graph
has grown since that run: the profile gained `sequence-proteinmpnn-genie3` on 2026-09-11, and the
fixture now declares 28 stages and plans 44. The
2026-09-05 section below records 36 structure pictures from that day's run. Picture counts move when
the fixture profile changes, so treat neither as a number a fresh run must reproduce.

A second generator arm runs. A fixture campaign carrying RFdiffusion3 alongside RFdiffusion
completed 45 stages, and its normalized pool contains an RFdiffusion3-origin candidate whose parent
belongs to that arm.

Paid dispatch from inside Claude Science completed on 2026-09-08. Campaign `pdl1-fal-supplied-n1`
committed 16 of 16 stages from a hosted session through the installed skill, reporting `state:
completed` and `ok` true, with none skipped and none resumed, 19:24:45 to 20:20:20 UTC. The shell
run `pdl1-fal-supplied-n1-v2` executed the same plan earlier that day, and the two replicate to
within 0.16 percent on rank score, ipSAE and sc_DockQ. Use the guarded dispatcher for paid Modal
work. The synchronous `claude-binder execute` surface cannot receive Claude Science's Modal
completion notifications.

## What the free route completed, 2026-09-05

The free 43-stage graph completed inside Claude Science through the installed skill. A hosted
session composed, checked, materialized, ran preflight, and executed the whole plan with no stage
filter, so terminal validation ran. It reported 43 of 43 stages, state `completed`, `ok` true, 43
receipts, 36 structure pictures, zero receipts carrying a spend record, and zero provider-facing
calls. Those figures were re-derived by walking the receipts directly rather than read from the
session's summary. This is evidence about the graph, the transport, the receipts and the renderers.
It is not evidence about any provider route or any scientific result.

The optimization controller decides a single-target round. A single-target round delegates its
stop decision to the controller, and a stopped round gets its own copies of the sequences and poses
it carries, written under that round's output directory. That applies to any campaign whose
optimizer stops early. A multi-target round still takes the older path, because folding several
targets into one decision is a choice about the campaign rather than a mechanical one.

## What the guarded Modal route completed, 2026-09-04

The guarded Modal route has completed a paid campaign end to end. That campaign committed all
16 stages, including `output-check` and `render-viewer`. All three scientific provider stages
succeeded, all artifacts returned, four structure pictures rendered, and every sandbox closed. The
run recorded 622 seconds of provider-job wall time and 4.2837978 USD of admission-control estimates
under its 5 USD cap. Modal supplied no per-job settled amount, so the actual bill remains unknown.

The terminal report that campaign wrote is `ok: false`. Its 24 findings came from applying
optimization-only lineage requirements while `optimization.enabled` was false, not from a failed
stage, receipt, or artifact. The current package gates those checks on explicit optimization and
tolerates only insignificant float-to-float replay drift. A read-only replay against that
run root now validates all 16 receipts and artifacts with zero errors.

The supplied-candidate contract dry run reaches fourteen downstream stages offline when it reads a
completed artifact tree. Run it with an explicit source tree when replaying a new plan:

    PYTHONPATH="$BINDER_SKILL" python3 -m claude_binder.contract_dry_run \
      --plan <new_run_root>/run-bundle/run-plan.json \
      --artifact-root <completed_run_root>/artifacts \
      --from normalize-candidates \
      --fixture-root <scratch>/dryrun

`--artifact-root` and `--fixture-root` must name separate trees. The dry run makes no provider call.
It checks the supplied-candidate downstream route, not provider dispatch, artifact return, or
settlement. Alternate predictor, generator, and optimization routes remain explicit coverage gaps:
a stage without a runner reports a dry-run failure. Against that campaign's completed artifact
tree, `--from promote` checks all seven remaining stages through `render-viewer` with zero findings
and zero provider calls.

The executor follows graph dependencies, not a prefix counter. A dependency-ready stage can finish
and checkpoint before an unrelated earlier stage. A requested stage with an unmet predecessor
refuses before it runs. On `--resume`, valid completed receipts are reconciled into
`stage-checkpoint.json`. A fully completed, unfiltered `execute --resume` performs terminal
validation without provider preflight or provider work. It still validates the bundle, receipts,
artifacts, and terminal report.

## HER2 cross-provider canary, 2026-09-04

A second target now has pose-recovery observations. HER2/ERBB2 extracellular constructs were folded against the
crystallised affibody Z<sub>HER2:342</sub> (PDB 3MZW, 58 aa) plus two challenge controls, on fal and
Modal, with **ESMFold2-Fast** on both arms and both resolving
`biohub/ESMFold2-Fast@b28d8ace5e05e61e5bec1e6820cfd3e221819d12`. Unlike the PD-L1 comparison, this
one does not confound predictor with provider.

Pose recovery depended on the target construct. The affibody epitope computed from 3MZW spans
HER2 372 to 517, which crosses the domain III boundary. At a 211-residue target window the crystal
binding mode is recovered on both providers: sc-DockQ 0.859377 on fal and 0.828754 on Modal, fnat
0.800 and 0.733, interface RMSD 0.782 and 0.821 Å. At 161 and 146 residues the same binder scores
sc-DockQ 0.0115 to 0.0123 with **fnat 0.000** and ligand RMSD near 50 Å, while ipTM stays at a
middling 0.33 to 0.47. All three windows contain every recorded epitope residue:
372–517, 365–525, and 320–530. The failed constructs therefore retained the complete contact span.
The observations support testing surrounding structural context; they do not identify the cause of the failed poses.

Both challenge controls returned **ipSAE exactly 0.000** at every tested window on both providers. The
wild-type Z domain has an identical scaffold and length, with twelve substitutions on the binding
face, and both controls received low scores. This supports discrimination within this panel.
The record supplies no experimental non-binding measurement for the wild-type scaffold.

Cross-provider agreement at the 211-residue window is 1.7 percent on ipSAE (0.468219 fal against
0.476111 Modal) and 0.4 percent on ipTM. Modal returned slightly higher confidence on all seven
complexes.

A three-seed follow-up on Modal found the separation seed-stable: positive ipSAE spans 0.010719
across seeds 0, 1 and 2, sc-DockQ stays in the high-quality band at every seed, and both negatives
hold ipSAE 0.000 at all three. No row here is a seed-sensitive review case. Fixed seed is
nonetheless not bit-reproducible on this route: two seed-0 folds of one complex differed by 8.67e-05
in ipSAE and 2.98e-03 Å in interface RMSD.

Two limits on the claim. HER2 has no calibrated control panel of its own, so the PD-L1 ipSAE gate
of 0.4245905 remains an observation about method behaviour and not a HER2 winner rule. The positive
clears that value at every seed, and every negative misses it by its full value. Recovering a pose from
a complex deposited in 2010 is a control on the pipeline, not evidence of prospective performance on
novel binders; training-data overlap was not determined and cannot be excluded.

The run exercised the packaged fal client, the campaign-specific Modal script, and package scoring
(`pae_matrix`, `derive_chain_mapping`, `compute_ipsae`, `compute_dockq`). It did not exercise
compose, check, materialize, execute, the spend ledger, the approval gate, or the guarded Modal wave
dispatcher. Modeled spend was 5.8130 USD on fal and 2.9138 USD on Modal; neither is settled.

## Historical guarded Modal checkpoint, 2026-09-02

An earlier guarded Modal run reached eight of sixteen stages. The later campaign that reached
`render-viewer` through the guarded Modal route supersedes it. One fact from that checkpoint still
applies: every materialized bundle pins `lane.py`, so a package repair forces a fresh run.

By 2026-08-31 the recorded runs were one local contract campaign, one live
cross-provider PD-L1 co-fold comparison, and one extended Modal counter-screen.

The local-contract campaign completed all 43 stages through the placed Claude Science skill. It
used fixture adapters, made no provider call, and recorded no spend. This run verifies the campaign
graph, receipts, ranking, reporting, and artifact surfaces.

The live campaign folded 20 de novo PD-L1 designs and two controls with ESMFold2-Full on Modal and
ESMFold2-Fast on fal. Modal completed 22 folds through the Claude Science host compute surface. fal
completed 22 folds through the packaged client from the workstation. The two ipSAE rankings have a
Spearman correlation of 0.931 across the 20 designs.

Designs 005, 006, 009, and 020 clear the recorded seed-0 control-relative ipSAE and zero-clash
conditions on both predictors. The extended Modal run re-folded those four designs over seeds 0,
1, and 2 against PD-L1, PD-L2, CD80, and CD86. All four have positive selectivity deltas against
every off-target. Designs 005, 006, and 020 satisfy the run's strict rule that the seed supplying
maximum on-target ipSAE must also be clash-free. Design 009's maximum-ipSAE seed has one clash and
beats a clash-free seed by 0.000456 ipSAE. Keep it as a non-robust review candidate rather than
interpreting the rule failure as evidence of no binding or no selectivity. Only designs 005 and
020 pass every recorded condition on every seed. Design 006 passes the argmax rule but fails one or
more conditions on seeds 1 and 2. Treat 006 and 009 as different forms of seed sensitivity.

No candidate has been synthesized, expressed, or assayed. The panel covers the annotated IgV
domains of PD-L2, CD80, and CD86. It does not establish whole-protein or cellular specificity.

That August record defined the execution boundary at the time. The Claude Science session ran
Modal through `host.compute.create` with a campaign-specific ESMFold2 script and used package
reducers after harvest. It did not test the guarded transaction path. The packaged fal client
completed the same seed-0 inputs from the workstation. A later fal call returned no result before
the package's selected wall bound; its remote state and settled cost remain unknown. Do not treat
that selected duration as a provider limit or a sequence-length rule.

Direct Modal stages through `claude-binder execute` remain disabled. The guarded Modal wave
dispatcher is the supported paid package path for a materialized and approved plan. Earlier guarded
runs established its intermediate transaction boundaries. The completed campaign carried the
remaining path through the final local stages and viewer.

The guarded dispatcher owns artifact return, promotion, and campaign commit. The completed campaign
verified submission, completion, receipt validation, returned file hashes, checkpoint advancement,
local continuation, and viewer output. Reattachment remains covered by tests and earlier targeted
live evidence rather than by a retry inside it.

The package includes `supplied-candidates-modal.template.json`. It accepts a complete candidate
manifest, bundles and verifies the named FASTA and structure files, and skips generation and
sequence design. The completed Modal campaign proved this profile through the guarded paid route.

Qualify a Modal arm against a campaign's own target with `lane qualify --receipt`. `runtime-check`
refuses an unqualified Modal arm, and the qualifier runs its canary through `subprocess`, which
reaches no Modal job surface, so the receipt route is the one that closes the loop. It builds the
roster row from a receipt a session harvests. One canary has run it end to end and produced a `PASS`
Modal roster row. See [Qualify a Modal arm](modal-qualification.md).

Modal settled costs are readable. `WorkspaceBillingReport` is a unary-stream RPC, so calling it as
a unary-unary call blocks until the timeout. Iterating it returns per-app, per-hour rows carrying a
decimal `cost` string. The report lags by hours, so a job that has just finished still has no settled
figure and its cost stays an estimate until one appears.
