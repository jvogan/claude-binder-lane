# Design and filters

Read this before the first generation wave for a target. It covers target biological research, the structure-method backbone floor, and pre-scoring filter requirements.

## Contents

- [The target dossier](#the-target-dossier)
- [The structure-method floor](#the-structure-method-floor)
- [The target-fold mimic](#the-target-fold-mimic)
- [Genie3 design constraints](#genie3-design-constraints)
- [The supplied-candidate manifest row](#the-supplied-candidate-manifest-row)
- [Pre-scoring filters](#pre-scoring-filters)

## The target dossier

Before generating or scoring designs for a target, record four properties: oligomeric state, constitutive cofactors and bound partners, construct boundaries (including residue range and fusion context), and deposited structures showing the epitope in an ordered state.

The scoring construct matches that system. When the dossier flags a cofactor as fold- or interface-required, at least one ranking instrument must represent it. Freeze the per-arm construct at validation gate time and use it across all seed tiers.

Verify cofactors directly from the output structure. The scoring job parses the output structure and compares chain counts and cofactor atoms against the frozen construct. On a mismatch, it records score as NaN with a construct-failure status.

## The structure-method floor

Every floor-bound structure-design method contributes at least 50 backbones to every target scored pool, recorded per structure method in the design-count ledger. Seven methods carry that floor in the published roster: RFdiffusion, RFdiffusion3, FreeBindCraft, BoltzGen, PXDesign, Proteina-Complexa, and Genie3.

Recompute the method-by-target matrix every governor cycle, write it to a state file, treat every cell below 50 as an open obligation, and prioritize under-represented methods. A missing matrix is a process deviation. A method that passed bring-up and contributed zero ranked backbones campaign-wide is a defect.

Beyond that 50-backbone floor, allocate generation compute toward the best-performing methods for each target and epitope.

### The target-fold mimic

A structure generator conditioned on the full target complex, at a requested binder length close to a target chain, can reproduce that chain fold. In training data, a protomer binds its oligomeric neighbors. This failure occurs most frequently on homo-oligomeric targets. Novelty filtering detects target-fold mimicry before scoring.

## Genie3 design constraints

This section summarizes notes from a recorded Genie3 campaign. Treat the quality and secondary-structure observations as results from that run, rather than enforced limits or published validation. Binder-mode output files are C-alpha traces, as described in the [tool catalogue](tool-catalogue.md).

### Length, as a quality observation and a selected-tool constraint

The [Genie3 adapter](../claude_binder/adapters/genie3_generator.py) accepts any positive ordered binder-length range. It rejects `args.binder_length_min < 1 or args.binder_length_max < args.binder_length_min` with `binder length bounds N-M are empty`. The recorded quality observations do not establish an upstream minimum-length requirement.

Short output is a quality problem, and the note documents it. Genie3 inherits its training set from Genie 2 and Genie 1: AFDB structures filtered to pLDDT above 80, a set with effectively no entries below 50 residues. The model never saw a single-domain backbone that short. Below roughly 50 residues the note records degenerate output, short helices and kinked snippets with poor scrmsd, rather than a failure the run would notice. Heed that observation when you choose a binder length. The note also reads `min_length: 50, max_length: 50` in the upstream `examples/unconditional/experiment.yaml` as a published floor. Those two fields set one example's sampled length, and no file here shows them rejecting anything.

Binder accepts any positive ordered length range. It does not impose a package-wide
50-to-120 window. Resolve a selected tool's actual limits from its runbook or
service contract, then record any narrower limit with the route. `generator_preflight.py`
checks only positivity and ordering. `adapters/genie3_generator.py` passes both
bounds to the BinderBench problem JSON and checks the returned chain against the
request.

### The best length regime

Genie 2 measured designability declining with length and diversity declining with
length, and Genie3 inherits both. The note's comfort zone is 60 to 150 residues.
The prior campaign chose 60 to 90 residues; that is an observed design choice,
not a Binder ceiling.

### Helix bias

Genie3 favours helices. Of 10 unconditional samples at length 150, 4 reached 10 percent or more beta strand and the remaining 6 were near-pure alpha bundles. That is the note's own recorded observation, taken in unconditional mode rather than binder mode. The stated cause is that AFDB under-represents strand content against the PDB. An alpha-helical miniprotein antagonist gains from the bias. A design that needs a beta sheet at the interface should use a different generator.

### Sampling knobs

`direction_scale` steers the sampler toward the site residues. The note gives 2.0 for binder mode, 0.8 for unconditional at or below length 300, 0.0 for unconditional above 300, and 0.1 for motif scaffolding. `adapters/genie3_generator.py` defaults `--direction-scale` to 0.0, which is unsteered. A binder phase that wants the note's binder-mode setting passes `--direction-scale 2.0` explicitly.

`n_sample` has a floor of 50 for a binder bake-off. The note records a run at an n of 12 and treats its result as a variance artifact. The adapter fills `n_sample` from `--count`, so the phase count is that knob. [The structure-method floor](#the-structure-method-floor) names the same number for an unrelated reason, and the two floors are not interchangeable. The roster floor counts backbones that reach a target scored pool, and pre-scoring filters drop some of a phase, so a phase of exactly 50 satisfies the sampling floor and can still leave the roster floor open.

`sampling_steps` is not exposed in the stable YAML. The default schedule stands.

### Genie3's own evaluator thresholds

Genie3 ships a Version 0 binder filter set inside its own `evaluate` stage: `complex_scrmsd` below 2.5 Angstrom, `binder_ptm` above 0.8, `min_interface_pae` below 1.5 Angstrom, and hotspot coverage at or above 0.8. Passing complexes land in `v0_success/successful_complexes/`.

This package neither adopts those thresholds nor runs that stage. `adapters/genie3_generator.py` builds `genie3 generate`, which does not evaluate, and offers no way to build `genie3 run`. No Genie3 filter fires on any candidate in this lane. Promotion thresholds live in `campaign.json` and in the validation gate, and they share nothing with the four values above. Record them only for reading a Genie3 output directory somebody else produced.

## The supplied-candidate manifest row

A profile whose generator sets `source: "supplied"` reads a JSONL manifest, one candidate per line.
`candidate_normalizer.REQUIRED_ROW_FIELDS` refuses a row missing any of these 24 fields, naming the
ones it could not find.

| Group | Fields |
| --- | --- |
| Target | `target_id`, `target_sha256` |
| Identity | `candidate_id`, `parent_candidate_id`, `status` |
| Provenance | `origin_generator`, `generator_mode`, `sequence_designer`, `generator_seed` |
| Sequence | `sequence_path`, `sequence_sha256`, `sequence_length` |
| Structure | `structure_path`, `structure_sha256`, `design_pose_path`, `design_pose_sha256`, `residue_map_sha256` |
| Optimization | `optimization_round`, `last_optimizer` |
| Diversity lineage | `root_backbone_id`, `tm90_cluster_id`, `structure_method`, `seq_method`, `fold_class` |

`generator_mode` is `backbone-only` or `sequence-structure-codesign`. `status` is one of
`generated`, `sequence-designed`, `filtered`, `promoted`, `failed`. Paths are relative to the
manifest, and materialization bundles each named file and verifies its declared hash.

A supplied candidate rarely arrives with the diversity lineage group. This lane runs no TM-score
clustering stage and no fold classifier, so `candidate_lineage.backbone_lineage` records the
conservative values for its own de novo backbones, and a supplied row may use the same:
`tm90_cluster_id` set to the candidate's own id, which states a singleton cluster, and `fold_class`
set to `unknown`, which records that classification was not performed. Both stay explicit so
promotion cannot read missing provenance as diversity. Supply a real cluster id or fold class only
when you measured one.

## Pre-scoring filters

Five checks run before co-folding spend. `AVAILABLE_FILTER_IDS` in `claude_binder/filter_contracts.py` names those five, and two more are retired for missing executables. Record every rejected design ID and verify its absence from downstream scoring pools.

No filter runs in the generation container. The package screens candidates in two separate CPU stages, `filter-integrity` and `filter-novelty`, which all 26 shipped profiles carry and no shipped profile spells `filter-screen`. Both run `claude_binder.adapters.screen_filter`, which reads the structures and model scores the generator already wrote and makes no provider call (`claude_binder/adapters/screen_filter.py`). That adapter stamps its own report with `stage_id` set to `filter-screen`, so the name in the report is not a stage you can schedule or find in a composed campaign. That stage offers five filter contracts by ID: `composition`, `exact_duplicates`, `liability_chemistry`, `sequence_novelty`, and `model_likelihood` (`claude_binder/filter_contracts.py`). Two more are retired, `structure_novelty` for missing TM-align and Foldseek and `secondary_structure` for missing DSSP (`claude_binder/filter_contracts.py`). Declare filter contracts explicitly in `campaign.json` under `filters.contracts`.

Novelty filtering rejects a candidate matching any of these criteria:

- Greater than 60% sequence identity over greater than 50% coverage against UniRef90 or the known-binder corpus.
- At least 30% gapped local identity over at least 40 aligned residues.
- TM-score of at least 0.5 to any chain of the target reference structure or positive-control complex.

Compare candidates against UniRef90, the known-binder corpus, and every chain of the target reference structure and control complex. Detect ubiquitin variants with local sequence alignments.

Liability filtering checks cysteine parity, homopolymer runs, and surface hydrophobic patches.

The shipped standard-library filter applies these configurable defaults before co-folding:

- Reject odd cysteine counts. The protocol requires cysteine parity, and sequence alone does not resolve disulfide pairs, so an unpaired cysteine is treated conservatively.
- Reject homopolymers longer than four residues (`filters.liability_rules.maximum_homopolymer_run`).
- Reject a 12-residue window containing more than seven residues from `AVILMFWY` (`filters.liability_rules.maximum_hydrophobic_residues` and `hydrophobic_window_length`).

The local target-chain mimic source compares every design with every protein chain parsed from configured target structures and enabled positive-control structures. It reports the highest-identity ungapped window. The maximum identity cutoff is 0.60. The default window length is 40 residues (`filters.metric_sources.sequence_novelty.window_length`).

No general known-binder corpus ships. Such a corpus needs a versioned FASTA or JSONL file identifying binder sequences, targets, experimental evidence, source accessions, licenses, and SHA-256 digests. The repaired PD-1 control supplies one positive-control sequence and does not constitute a known-binder corpus. Provide positive-control sequences and target structures in the campaign configuration.

Structural target-chain comparison needs a pinned structure-comparison executable in the environment (`foldseek`, `USalign`, `TMalign`, or `TMscore`). Without one, use local sequence novelty metrics.

Monomer foldability folds the binder alone and applies a per-target mean pLDDT threshold frozen at the validation gate. The default threshold is 70 on a 0 to 100 scale (0.7 on a 0 to 1 scale).

Structural plausibility checks backbone geometry, steric clashes, and core packing.

Specify structural-plausibility thresholds at campaign kickoff, and define them explicitly in `campaign.json` under `filters.contracts`.

Redundancy filtering clusters de novo pools at 90% sequence identity before scoring. For mutagenesis and close-variant pools, drop exact-sequence duplicates before scoring and enforce diversity during candidate selection.

Sequence design restraints use soluble variants of sequence-design models when generating or selecting designs. Base variants apply to backbone search. Apply the local composition perplexity penalty as a restraint against homopolymers and record its score for each sequence. Compute sequence-model log-likelihood for every ranked design.
