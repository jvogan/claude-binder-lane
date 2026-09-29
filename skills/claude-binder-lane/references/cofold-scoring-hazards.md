# Co-fold scoring hazards

Check chain identity, parsed structures, reference poses, and required model
outputs before scoring a new route. The observations below explain failures
that produced plausible scores in earlier runs. The final sections address
score independence and interpretation across predictors.

## Resolve chain roles from the structure, never from a constant

Chain identifiers are not stable across predictors. Measured on one PD-L1
design: one ESMFold2 route writes the binder as chain A and the target as chain
B, and the other writes the reverse. Boltz-2 and OpenFold3 each have their own
convention.

`ipsae_min` is invariant to the swap. On the same file the directional pair
reads 0.616123 and 0.705266 one way and exchanges the other way, so the minimum
is unchanged and an ipSAE-only pipeline never notices. The other scorers do notice, and fail quietly:

- `compute_dockq` with a fixed `{"target": "A", "binder": "B"}` map returns
  `mapping_status: no-reference-interface` on every row of the inverted arm.
- `compute_site_metrics` takes chain-qualified site labels such as `A:9`, so a
  site contract written against chain A silently scores the wrong chain when the
  target is chain B.

Resolve the pair from the target sequence, residue map, and structure provenance.
Validate a configured chain against that identity. Residue counts can confirm
a mapping but cannot establish it when lengths coincide or constructs differ.
Record the mapping on the scored row so the direction of each stored score is
clear. Resolve an ambiguous mapping before scoring.

Apply the same check before sequence design. A backbone generator can rename
the input target chain. Map its output back to the recorded target and verify
that the intended target remains fixed in the designed result.

`ChainOrderWarning` already exists in `binder_metrics` and is the right place to
make a mismatch loud. A ranking script that scores ipSAE alone passes this
hazard undetected and hands its chain assumption to whatever scores sc_DockQ
next.

## Normalise mmCIF before parsing it

`parse_structure_atoms` reads one `_atom_site` row per line. Multi-line row
wrapping is legal mmCIF and at least one hosted predictor emits it, so a
186-residue complex parses as 12 residues **with no error raised**. Measured
across ten such structures, every one mis-parsed. The files were full size, 135
to 176 KB with 1455 atom records for one of them, and the column list was the
standard 19 `_atom_site` fields.

The metrics accept the mis-parsed result.
`compute_dockq` returned `mapping_status: ok` on the mis-parsed structures and
`compute_site_metrics` returned a plausible IoU. Nothing in the scored row
reveals that 174 of 186 residues were dropped.

Tokenize the loop by field count rather than by line: read the `_atom_site`
column headers, then accumulate whitespace-separated tokens from the rows that
follow until `len(columns)` are available, emit one row, and repeat until the
loop ends. Leftover tokens at the end mean the file is malformed and should
raise. The parser should additionally assert its residue count against the
distinct `(chain, seq_id, ins_code)` triples the atom records imply, so a
mis-parse fails instead of scoring.

A predictor that writes one row per line is unaffected, which is why this stays
hidden until a second predictor is added.

## The sc_DockQ reference is the design pose

In one supplied-candidate PD-L1 cohort, the shipped design pose was
coordinate-identical to that arm's own co-fold prediction: maximum absolute CA
deviation 0.0 A over all 186 residues, on every candidate that shipped a pose.
Any sc_DockQ computed against such a reference returns exactly 1.0 with fnat 1.0
and RMSD 0 -- a self-comparison carrying no information -- and it enters the
ranking as a perfect score.

The protocol's sc_DockQ term compares a co-fold against the **design** pose from
the generator and sequence designer. Supply that: the backbone with the designed
sequence on it. If no design pose exists, score one predictor's co-fold against
another's and label it cross-arm agreement. A co-fold scored against itself is no
substitute.

## Verify PAE availability on the selected deployment

The hosted BioNeMo endpoints observed on 2026-09-11 returned no PAE matrix.
That limits the scores available from those responses. It does not establish
a limit for every hosted deployment of either model. ipSAE requires PAE;
scalar confidences and returned coordinates cannot reconstruct that matrix.

Check the selected deployment's actual output before scaling an ipSAE arm.
Verify that the matrix matches the returned structure's residue order and
dimensions. Use any authorized route that supplies the required artifacts.
Self-hosted Boltz with full PAE output is one recorded option. See
[NVIDIA BioNeMo NIM route](bionemo-nim-route.md) for the deployment observations.

## A generator's own ranking score can be the screening metric under a different pair mask

Added 2026-09-21, when BindCraft2 was catalogued. This one is not a bug in the
scoring path. It is a reading error the scoring path cannot stop.

BindCraft2 ranks its accepted designs on `i_pDAE`, and `i_pDAE` is ipSAE with
the pair mask swapped. `bindcraft/filters.py` selects the residue pairs whose
C-alpha atoms sit within 8 angstroms, takes d0 from each residue's own partner
count, scores each pair with `1 / (1 + (PAE / d0) ** 2)`, averages over that
residue's partners, and takes the maximum over residues and over both
directions. `binder_metrics._directional_ipsae` is the same thing with the pair
mask changed: Dunbrack selects the pairs by PAE below a cutoff, 10 angstroms
here. Those are different sets, not the same set named twice. A partner with low
PAE outside the 8 angstrom shell moves ipSAE and leaves `i_pDAE` fixed, and a
contact whose PAE sits at or above the ipSAE cutoff moves `i_pDAE` and is absent
from ipSAE. Since d0 comes from the selected-partner count, the distance scale
moves with the mask. Three smaller differences: `i_pDAE` reads coordinates and
resolved flags where ipSAE reads no distance, `i_pDAE` pools every binder chain
into one block, and `ipsae_min` reduces the two directions by the minimum where
`i_pDAE` takes the maximum. The two d0 helpers are bit-identical at 45 of the 60
partner counts from 1 to 60 and one unit in the last place apart at the other 15,
because upstream raises to `** (1 / 3)` where the lane calls `np.cbrt`, with a
largest gap of 8.9e-16. d0 sits on its 1.0 floor until a residue has 27 partners,
so on an ordinary binder interface both are the same fixed
`1 / (1 + PAE ** 2)` kernel and the mask is the whole difference for a single direction.

**The two directions are not the same difference.** BindCraft2 returns one
binder-to-target PAE block and its transpose, so its two directions are the same
numbers read two ways. `compute_ipsae` slices `matrix[target, binder]` and
`matrix[binder, target]` independently, and AlphaFold PAE is asymmetric, so the
lane's two directions are two different measurements. `ipsae_min` is then the
smaller of two measurements where `i_pDAE` is the larger of one measurement seen
from both sides. Read the shared construction as holding per direction, not for
the reduced value.

Clearing `minimum_ipsae_min_ensemble` on a BindCraft2 row is not a second
construction agreeing with the first. It reaches the gate three reductions deep, as the
minimum over the two directions, then the seed aggregation, then the arithmetic
mean across predictors, so a single structure's number is not what is compared.
Calling it more conservative would be wrong twice over. Minimum beats maximum
only on the same directional scores, and the mask, the partner counts and the
predictor all move between the two. And the published seed aggregate is `max`,
which `campaign.template.json` ships as `"seed_aggregation": "max"`, so the
value that reaches the gate is the optimistic seed. With one predictor enabled
it is one structure's `ipsae_min` at the best seed.

The independence that survives is the predictor. The section below is the only
measurement of what that is worth and it does not carry much weight on its own:
eight designs, the same ipSAE implementation on two predictors, Spearman
-0.1429 at p = 0.736, one sample per fold and no seed controlled. That section
says to measure the seed band before attributing the spread to the predictor,
and nothing has.

`minimum_sc_dockq_ensemble` and the clash ceiling are a different construction,
though not an unrelated one. Both read predicted geometry, sc_DockQ's reference
is the design pose, which on this route is the accepted BindCraft2 complex
itself, and `maximum_clash_count` is taken at the seed the `ipsae_min` aggregate
already selected.

Nothing here establishes that the generator optimised against the lane's screen.
Shared construction shows the two metrics are near-relatives. It does not show
that selection on one transferred to the other through a different predictor, and
nothing here has measured that. Treat the dependence as a reason
to read a passing screen carefully, not as a measured effect.

The ranking formula depends on the claim mode. `ranking_policy.ranking_formula` returns
`(ipsae_min_ensemble + sc_dockq_ensemble) / 2` only for the four reduced
candidate-claim modes in `REDUCED_RAW_MEAN_MODES`. A published claim ranks on
the z-scored weighted formula `lane._ranking_formula` builds from the campaign's
own configured metrics and weights, and `campaign.template.json` ships those
weights as `ipsae_min_z` 0.8 and `sc_dockq_z` 0.2. So the raw mean is where one
of two equal terms is a near-relative of the generator's own estimator, and on
the shipped published weights that near-relative carries four fifths of the
score.

**The lineage gate does not report this, and the reason is worth knowing.**
`ranking_policy.ranking_mode_selection` already refuses a published ranking
claim whose enabled modes all come from one lineage, against
`PUBLISHED_RANKING_MODE_LINEAGE_FLOOR = 2`. That gate exists to stop this class of
false agreement, and it does not reach this case, for two reasons.
It enumerates `cofold.predictors` only, so a generator is outside its scope. And
lineage is a co-folding model's embedding family, read from the hard-coded
`arms.PREDICTOR_LINEAGE_IDS`, so it counts model families and never inspects a
metric's construction. A row ranked on `i_pDAE` and screened on `ipsae-min`
clears it untouched.

The general rule: before treating a screen as an independent check on a
generator, read what the generator ranks on. A generator
that selects on a near-relative of the screening metric may already have
optimised against part of the screen, whatever the two are called.

This adapter does record its own: every published row carries
`selection_metric`, `selection_metric_value` and `selection_metric_note`. What
no code does is compare that field with the lane's screening metric, so the rule
is enforced by a reader and not by a check. `i_pDAE` is the first case in the
catalogue, and the BindCraft2 row's own `gate.conditions` carries it too.

Every statement here about BindCraft2 itself was read from an upstream checkout
this repository does not contain, so a reader here cannot check it: the 8 angstrom C-alpha mask, the clamp of
the partner count at 19, the maximum over both directions, the coordinate and
resolved-atom gate, the chain pooling and the absence of PyRosetta all live in
`bindcraft/filters.py` and its neighbours. The lane side is checkable here and
that is the half the tests freeze. Committing one accepted complex and one
`campaign_metadata.json` from a real campaign under
`claude_binder/data/roster-evidence/` would close the gap, and until it is closed
read the upstream half as a dated reading rather than as evidence.

## What a cross-predictor score licenses

Two results from that run bound the interpretation of any second arm.

A threshold calibrated on one predictor does not transfer to another. The
measured PD-L1 gates were calibrated on 30 rows of an ESMFold2 single-sequence
route. On a self-hosted Boltz-2 arm the positive control clears the recorded
`positive_control_minimum_ipsae_min` where the ESMFold2 route fails it. The
panel then satisfies the recorded numbers without being a validated panel. A new
predictor needs its own control rows before its scores gate anything.

A ranking can also fail to survive the substitution. Over eight designs,
Spearman between the ESMFold2 two-arm mean ipSAE and self-hosted Boltz-2 ipSAE
was -0.1429 (p = 0.736) -- same metric implementation, same target, same
sequences. No seed was controlled in that comparison and each fold drew one
sample, so sampling noise is not separated from the predictor change. Measure
the seed band before attributing a cross-predictor difference to the predictor.
