# Scoring details

Read this before writing any scoring code, and again before the validation gate.
`SKILL.md` carries the two scores and the ranking rules. This file carries the
settings, the traps, and the cases those rules do not cover.

## Contents

- [Choosing a positive control](#choosing-a-positive-control)
- [Calibration bands](#calibration-bands)
- [Known deviation: the floor ships, the ceiling is deferred](#known-deviation-the-floor-ships-the-ceiling-is-deferred)
- [What the release specifies](#what-the-release-specifies)
- [Chain identity in predicted files](#chain-identity-in-predicted-files)
- [Why this score](#why-this-score)
- [Seed tiers](#seed-tiers)
- [A recorded seed does not pin the structure on every arm](#a-recorded-seed-does-not-pin-the-structure-on-every-arm)
- [Multimeric targets](#multimeric-targets)
- [Substitute arms](#substitute-arms)
- [The contact metrics are ours](#the-contact-metrics-are-ours)
- [Interface chemistry is a separate measurement](#interface-chemistry-is-a-separate-measurement)

## Choosing a positive control

Three rules govern control choice. Prefer a non-antibody protein control, such as a natural
ligand, a receptor ectodomain, a viral entry protein, or a peptide, because co-folding models
underperform on antibody-antigen interfaces. An antibody control is secondary: one that fails
separation is weak evidence and the gate passes on the fold check alone, while one that
passes is strong evidence. Published de novo miniprotein binders are excluded as separation
controls, because they were themselves likely selected by co-folding metrics and their
separation is circular.

### Calibration bands

Separately from the gate, establish a raw-score **ceiling band** from native
protein-protein complexes such as barnase and barstar, and a **floor band** from constructed
negatives: a non-interacting pair, a sequence-shuffled binder, a cross-pair mismatch. Run both
at the same seed count, chain construct, and pairing as the designs, with MSAs on both chains,
or in single-sequence mode where an arm has no MSA encoder.

These are raw-score calibrations only. Z-scores depend on the pool being scored, so
calibration runs report raw values and z-scoring is reserved for the design pool. Panel
members are native complexes from outside the campaign's own target set. Begin to consider a
target saturated only when its top designs' raw score is at or above the ceiling band on
the same model.

### Known deviation: the floor ships, the ceiling is deferred

The protocol calls for both bands. This skill ships the **floor band only**. The
ceiling band is deferred.

The ceiling has one stated use: a saturation signal that tells you a target has
stopped returning value for more spend. A run that scores an existing pool of
designs and generates none of its own has no allocation decision for that signal
to inform. On such a run the published per-target medians
in `released-record.md` are a stronger reference than a ceiling band: they compare
real designs against real published values, per design, on the same arms.

The floor ships because it is nearly free and it catches the failure this pipeline
is most exposed to. A broken generator produces zero signal and passes every check
that only tests for high scores. The floor band separates a genuinely hard target
from a broken instrument, and it supplies the negative-control maximum the writer
needs.

The ceiling band becomes required at the first run that generates designs and has
to decide when a target is done. Fix its trigger before that run starts.

### What the release specifies

The protocol leaves three scoring details open that the released re-score of the campaign
designs pins down. Use them as defaults and label them as the re-score's settings rather than
as the campaigns' own, because campaign-time values came from whichever predictors, settings,
and constructs each campaign chose at the time.

- **ipSAE**: the d0res variant at a PAE cutoff of 10 angstroms, computed in both directions.
- **sc_DockQ**: DockQ v2 over binder backbone atoms and target heavy atoms, with all target
  protein chains treated as one receptor, and identical protomers and binder copies assigned
  by the mapping that maximizes DockQ.
- **Target MSAs**: unpaired, one alignment per construct, the same alignment for every copy of
  a homo-oligomer, with the binder chain as a single sequence.

### Chain identity in predicted files

Chain order in a co-fold file follows that predictor's own input order and is not uniform
across arms. In the release most predictors write the target chains first and then the binder,
ESMFold2 writes the binder first, and Protenix v2 places the binder between the protomers for
part of the trimeric and dimeric designs. Identify chains by sequence, or by the chain IDs
carried in the PAE file, rather than by position. Any ipSAE or DockQ code that assumes chain A
is the binder returns wrong numbers on at least two of the three published ranking modes.

### Why this score

The three instruments were benchmarked as de novo binder filters on a published dataset of
3,532 designs against 13 targets. By macro-averaged precision, ESMFold2-Fast reached 0.62 and
ESMFold2 0.61, each above the strongest predictor reported for that dataset at 0.55, and a
per-target z-score ensemble of the three reached 0.66. Self-consistency DockQ alone is a
weaker filter at 0.45, and at the 4-to-1 weight it leaves discrimination unchanged at 0.65.
So the score measures co-folding confidence first and agreement with the designed pose
second. The DockQ terms are a check that the predicted pose is the designed one.

### Seed tiers

| Tier | Seeds per arm | Use |
|---|---|---|
| SCREEN | 1 | Fast iteration. Keep the top fraction the budget supports. |
| INTERMEDIATE | 5 distinct | Iteration between waves and optimization rounds. |
| FINAL | 5 distinct | Required for the sheet. |

Seeds are distinct integers per arm. Assert that the seed set size equals the recorded count,
and treat a per-design standard deviation of zero across more than one seed as a bug.

### A recorded seed does not pin the structure on every arm

Measured 2026-08-31 on `cofold-screen-esmfold2-fast` against a fal deployment. One completed
run folded `design-001` twice, with the same binder, the same target, seed 0, the same endpoint
and the same day. Both `predicted.cif` files were 153,082 bytes and their contents differed.
The seed argument does not pin the output on that arm, so re-running at the same seed does not
reproduce a structure and cannot produce a seed-robustness result. Boltz-2 takes no seed
parameter at all and behaves the same way for the same reason.

Size an ordering claim to that spread. In the same run the gap that decided first place was
0.002057 iPTM, and the same design folded twice differed by 0.002089. The top two designs were
therefore not ordered by that measurement, while ranks 3 and 4, separated by 0.13 and 0.16,
were. A gap narrower than your own replicate spread is not a ranking. Fold a design twice on
your own arm before you trust a close order, which costs one extra call and needs no new
tooling.

The recorded seed still identifies which call produced which row, which is what the receipts
use it for. On these arms it does not reproduce the structure.

The package makes that comparison itself, for the claim rather than for the order. For each
adjacent pair it records `rank_score_gap` and `rank_score_gap_spread`, the sample standard
deviation of the paired same-seed differences between those two designs, and marks the pair
`separable` only when the gap exceeds that spread. The leading tier runs down the order until
the first separable pair. When more than one design falls in that tier,
`best_design_claim_status` is `tied` and the report names no leader, with the reason recorded
in `best_design_claim_reason`. The ordered table still prints in full, so a close order is
display only and the claim field states the result.

`rank_score_spread` answers a different question. It is the sample
standard deviation across one design's own seeds, printed beside `rank_score_central_estimate`,
and it describes how noisy that one design is rather than whether two designs are apart. Treat
it as a floor rather than the whole noise, because it measures variation between seeds and the
same-seed variation described above is a separate source it does not capture.

Promotion to the five-seed tier happens **before** parents are selected for the next
optimization round. Each round's outputs are scored on the same instrument and seed count
used to rank the pool that seeded it.


### Multimeric targets

For an oligomeric target you may crop to a subset of protomers during design, but every
scoring and filtering prediction models the full multimer.

Compute ipSAE on the (n+1)-chain prediction with the binder mask as the binder chain's
residues and the target mask as the **union** of all other protein residues, re-deriving the
normalization terms from the union mask sizes. This single union-mask call replaces the
per-pair loop over binder and each protomer. Then take the minimum over the two directions,
the maximum over seeds, and combine across arms exactly as for a monomeric target. The
positive control is measured on the same construct with the same pairing.

Score every ranked design at both stoichiometries. **1:N**, one binder against all protomers,
is the construct that feeds the declared ranking score: `rank_zscore` for the published
three-mode method and the raw ensemble mean for a disclosed reduced candidate method. **N:N**, one binder per protomer at full
occupancy, is recorded as additional required columns: its `ipSAE_min`, its `sc_DockQ`, and a
binder-binder clash count. Disclose any design whose N:N score drops sharply relative to its
1:N score. Build the N:N reference by applying the target's own symmetry operators to the
single designed binder pose, and search the symmetric relabelings for the best mapping.

The protocol does not say how to reduce ipSAE across the N binder copies. The release scores
each binder copy against the union of target protomers and reports the **worst copy** as
`ipsae_min`. Adopt that and disclose it as the release's rule.

### Substitute arms

The published baseline permits one independent-lineage co-folder per unavailable or unvalidated
arm when the run records the deviation. The deployed ESMFold2-Full and ESMFold2-Fast modes share
one lineage, so neither fills the other's place as a substitute.

An AlphaFold2-Multimer-v3 adapter ships in this package, and the profile
`claude_binder/data/templates/profiles/afm-substitute-two-arm.template.json`
places it beside ESMFold2-Fast. No AlphaFold2-Multimer application is deployed. The
profile therefore keeps the adapter `operator-configured`, with its endpoint,
immutable source and weight revisions, qualified image, and remaining hardware and
network facts as operator requirements. `compose` can carry that contract, but
materialization and preflight refuse unresolved values, so an undeployed substitute
arm cannot enter a ranking silently.

The platform's **alphafold2** skill runs the AF2/AF2-Multimer evoformer outside this package.
It has no adapter here, so it does not fill the unrun arm either.

The preference order is fixed: **AlphaFold-Multimer-v3**, then **AlphaFold3 code running
OpenFold3 weights**, then **Chai-1**, then **Boltz-2 or Boltz-1**, all with target-chain MSAs.

The second entry names one specific build, the AlphaFold3 inference code run with OpenFold3
open weights. The release treats that as a separate predictor from the OpenFold3 package run
on its own weights, and says the two are numerically distinct. Building the OpenFold3 package
gives the other predictor, which the protocol does not list as a substitute arm.

The set of arms feeding the declared ranking score for a target is **frozen at validation-gate time** and
named on every row. Dropping a validated arm from a frozen mask is sheet corruption. Where a
reduced mask is disclosed, the raw mean runs over the realized terms only and the row names
them.

The published baseline treats agreement across its three configured modes as its strongest computational
evidence. The deployed ESMFold2 modes provide within-lineage variation. Record the unrun
published arm before interpreting the result.

## The contact metrics are ours

`site_contact_iou` and `target_contact_recall` are **this pipeline's own metrics**.
Neither name appears in the protocol, and neither has a published basis. Three
related numbers ship with them: `target_contact_precision`, `hotspot_recovery`,
and `offsite_contact_fraction`.

They are not gates. They do not decide whether a design is eligible, and no
threshold on them is required configuration. Their only ranking role is a tiebreak
after the declared ranking score, which costs nothing and orders rows that would otherwise be
ordered arbitrarily.

They are still computed, and reported beside every design. The protocol requires
every scored design to confirm at minimum that it engages the intended hotspot or
epitope, and it names no metric for that check. These five numbers are how this
pipeline satisfies that step. Dropping them would skip a required step.

A design with `target_contact_recall = 0.2` has failed nothing. The number
describes where on the target the design makes contact, and the rank key in
`SKILL.md` decides whether it ships.

Three reasons they stopped being gates.

1. **The published designs fail them.** At `target_contact_recall >= 0.5`, under the
   most favourable site selectable with hindsight, at most 50.9% of published designs
   survive, and two published designs have zero target contacts. A gate that removes
   half of the published designs measures the gate.
2. **No threshold exists to gate on.** Every shipping configuration carries a
   required-but-unset marker for both. Setting a number now means inventing one.
3. **The gate has no sanctioned relaxation path.** The protocol fixes the ladder at
   diversity caps, then liability flags, then novelty. A gate outside that ladder can
   block the mandatory 30 ranked rows with no disclosed way out.

The positive-control checks stay. Requiring a positive control to contact the
declared site checks the instrument rather than filtering designs. It is also the
closest this pipeline comes to the protocol's own control-separation condition.

Two positive-control thresholds are unset.
`positive_control_minimum_site_contact_iou` and
`positive_control_minimum_target_contact_recall` carry no value today. Set them from
the observed control run rather than choosing them in advance.

## Interface chemistry is a separate measurement

`ipsae_min` and `sc_dockq` do not read the chemistry of an interface. ipSAE reports how
confident the predictor is about the interface it drew. sc_DockQ reports how closely that
pose matches a reference complex. A design can hold a high `ipsae_min` across an interface
that buries almost no surface, and neither number falls when it does.

The `interface-biophysics` stage measures the interface itself. It reads the predicted
complexes the screen already wrote, so it starts no fold and makes no provider call. Enable
it with an `interface_biophysics` block on the campaign, and the executor builds the stage,
places it after `score-screen`, and publishes the table to `scores/interface-biophysics.jsonl`.

| Field | What it counts |
| --- | --- |
| `interface_buried_sasa_angstrom2` | Solvent-accessible surface lost when the two chains associate, summed over both sides |
| `interface_apolar_fraction` | Share of that lost area contributed by the elements the campaign names apolar |
| `hard_clash_count` | Heavy-atom pairs closer than the sum of their radii less the campaign's tolerance |
| `contact_density` | Interchain residue contacts per interface residue |
| `interchain_residue_contact_count` | Residue pairs inside the campaign's contact cutoff |
| `interface_residue_count` | Residues on either chain inside that cutoff |
| `hotspot_coverage_count` | Site residues among the contacted target residues, recorded beside `hotspot_count` |

**The campaign supplies every constant.** The stage holds no radius, no probe size, no
sampling count and no threshold. A run that omits one is refused rather than given a
default, because a constant chosen inside the package would travel onto a result row with
nothing behind it. `geometry.convention_source` is required and is copied onto every
measured row, so a table always states the convention that produced it.

```json
"interface_biophysics": {
  "enabled": true,
  "geometry": {
    "van_der_waals_radii_angstrom": {"C": 1.70, "N": 1.55, "O": 1.52, "S": 1.80},
    "apolar_elements": ["C", "S"],
    "clash_tolerance_angstrom": 0.4,
    "contact_cutoff_angstrom": 5.0,
    "sasa_probe_radius_angstrom": 1.4,
    "sasa_sphere_point_count": 960,
    "convention_source": "Bondi 1964 radii; Lee and Richards 1971 1.4 A probe; Shrake and Rupley 1973 sampling"
  }
}
```

Those values are one defensible choice that a campaign makes for itself. The radii
are Bondi's 1964 compilation. The 1.4 Angstrom probe is the water radius Lee and Richards used to define the
accessible surface in 1971. The sphere-point sampling is Shrake and Rupley's 1973 numerical
form of that surface, which is what `screen_geometry` implements. The 0.4 Angstrom clash
tolerance is the PROBE convention from Word, Lovell, LaBean and colleagues in 1999, which
MolProbity still uses: two non-bonded atoms clash when their van der Waals radii overlap by
0.4 Angstroms or more.

**Two of these constants have no single published value.** The contact cutoff is
the clearer case. Published interface work uses anywhere from 4.0 to 6.5
Angstroms between heavy atoms, and 4.5 Angstroms is common for calling a residue interfacial.
The 5.0 above falls inside that range and is a choice rather than a standard.
Moving it changes
`interchain_residue_contact_count`, `interface_residue_count` and `contact_density` and leaves
buried area and apolar fraction untouched, so a campaign comparing contact counts against
another study has to match that study's cutoff before the numbers mean the same thing. The
apolar element set is the second case: carbon and sulfur is the usual split in interface
work, this package cites no paper for it, and a campaign that counts selenium or halogens
differently should say so in `convention_source` rather than assume the reader knows.

**Reading the burial number.** Lo Conte, Chothia and Janin surveyed 75 protein-protein
complexes and reported 52 with standard-size interfaces burying 1600 plus or minus 400
square Angstroms in total across both sides. Halve the field to compare it against a
per-side figure. That survey is a reference range for judgement, and the package ships no
gate built on it.

**Gates are optional and each needs a source.** A campaign with no `gates` list measures
every prediction and removes none, which is the expected first shape. A gate names one of
the fields above, an operator, a threshold and a `threshold_source`. A gate on a field the
stage does not measure is refused, and so is a gate with no source. When at least one gate
is configured the stage also publishes `scores/interface-biophysics-observations.jsonl`.

**What this stage does not measure.** It counts no hydrogen bonds, no salt bridges, no
buried unsatisfied polar atoms, no shape complementarity, and no packing density inside
the binder. `hard_clash_count` compares the two chains against each other and not a chain
against itself, so a binder whose own core overlaps still passes here.
