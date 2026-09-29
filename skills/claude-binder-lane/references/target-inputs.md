# Target inputs

Use a known partner complex to derive the site when it fits the scientific
question. For a target with a UniProt accession, the helper writes the residue
map, campaign site block, reference complex, and site-resolution record.

For a synthetic construct without a UniProt accession, use `explicit-residues`
and retain the source structure, chain identities, residue map, and derivation
record. The helper's partner route requires an accession even for a local
complex. Do not invent one. A deposited complex can still support an explicitly
recorded site. Preserve the contact rule and any changes to the published
selection, including a different cutoff or an intersection across copies.

Every command below runs in the Python kernel with the session `PYTHONPATH` from [getting started](getting-started.md) step 6.

## Derive the site from a binding partner

Three inputs decide the site: the target's UniProt accession, the partner's name in words, and the target's author chain in the complex. Pin the RCSB PDB entry with `--pdb-entry` when you already know which structure you want.

```python
import subprocess
import sys

subprocess.run([
    sys.executable, "-m", "claude_binder.make_target_inputs",
    "--target-accession", "Q9NZQ7",
    "--partner", "Programmed cell death protein 1",
    "--pdb-entry", "4ZQK",
    "--chain", "A",
    "--out", "pdl1.residue-map.json",
    "--site-out", "pdl1.site.json",
], check=True)
```

That command prints this, with the campaign directory shown as `$CAMPAIGN_DIR`:

```
structure         $CAMPAIGN_DIR/pdl1.residue-map.reference-complex.cif
chain             A of ['A', 'B']
residues          115 (A:18 to A:132)
source_to_cleaned identity
site residues     22
residue map       $CAMPAIGN_DIR/pdl1.residue-map.json
site block        $CAMPAIGN_DIR/pdl1.site.json
resolution record $CAMPAIGN_DIR/pdl1.site.resolution.json
complex           4ZQK (experimental)
partner chains    ['B']
confidence        high
```

The 22 residues it derived for PD-L1 from the PD-1 chain of 4ZQK:

```
A:18 A:19 A:20 A:23 A:26 A:54 A:56 A:58 A:63 A:66 A:76 A:113
A:115 A:117 A:119 A:120 A:121 A:122 A:123 A:124 A:125 A:126
```

Four files come out of one run.

- `pdl1.site.json` is the `targets[].site` block. Copy it in whole.
- `pdl1.residue-map.json` is the identity map over the 115 residues of chain A. `site.residue_map_path` already points at it.
- `pdl1.residue-map.reference-complex.cif` is the complex the contacts came from. Set `targets[].structure_path` to this file. The residue map is the identity over this file's numbering, so a different copy of the same entry is not interchangeable.
- `pdl1.site.resolution.json` records the entry, the chains, the cutoff, the method, and a confidence level with its basis. `site.resolution_artifact_path` points at it.

Four things this route decides for you.

- **Cutoff.** The contact cutoff defaults to 5.0 Angstrom over heavy atoms, taken from the published campaign's in-silico record. Pass another finite positive value when the scientific protocol requires it. The site and resolution records preserve that value and label it as operator-supplied rather than attributing it to the default protocol.
- **Mode.** `site.mode` is `reference-partner-contacts`. `--site-mode` accepts that value and refuses the others.
- **Design residues.** `site.design_residues` defaults to the derived contact set, so this route leaves no `__REQUIRED__` placeholder. Pass `--design-residues` to widen or narrow the design region.
- **Confidence.** `high` means the accession matched one deposited entity, the partner was identified, and the entry reports an experimental method. Predicted coordinates give `medium`.

## Pin the entry when several complexes match

Leave out `--pdb-entry` and the helper searches RCSB. It refuses to choose among several complexes and names them:

```
make_target_inputs: RCSB PDB found 4 complexes for target accession 'Q9NZQ7' and
partner 'Programmed cell death protein 1': 5IUS, 3BIK, 3SBW, 4ZQK. Supply
--pdb-entry to pin the biological state before deriving a site.
```

Read those entries, choose the biological state you want to design against, and rerun with `--pdb-entry`. Different entries of the same pair can carry different constructs, mutations, and resolutions.

## Use a complex you already hold

Pass `--complex-structure` for a complex you have verified yourself, or when the session has no egress to RCSB. This route bypasses the search and still records provenance.

```python
subprocess.run([
    sys.executable, "-m", "claude_binder.make_target_inputs",
    "--complex-structure", "4zqk.cif",
    "--complex-kind", "experimental",
    "--pdb-entry", "4ZQK",
    "--chain", "A",
    "--partner-chain", "B",
    "--target-accession", "Q9NZQ7",
    "--partner", "Programmed cell death protein 1",
    "--out", "pdl1.residue-map.json",
    "--site-out", "pdl1.site.json",
], check=True)
```

`--chain`, `--partner-chain`, `--complex-kind`, `--target-accession`, and `--partner` are all required here. Repeat `--partner-chain` for a partner with several chains. `--complex-kind predicted` records that the coordinates are computed and lowers the recorded confidence to `medium`. The file is accepted as `.cif`, `.mmcif`, `.pdb`, or `.ent`, and the copy it writes keeps your file's format and suffix.

Both routes produced the same 22 residues from 4ZQK, from the mmCIF and from the PDB copy of the entry.

## Write the dispatched structure as PDB

A provider can refuse an mmCIF target and give no reason. On 2026-09-16 the RFdiffusion3 deployment on fal answered `HTTP 500` with `{"detail": "Internal Server Error"}` on a 115-residue PD-L1 target supplied as mmCIF. The identical coordinates written as PDB produced backbones on the next attempt. Both files carry 874 atoms with the same atom names, residue names, chain B, author numbering 18 through 132, and coordinates agreeing to three decimals.

The response names no cause, so which part of the file the app rejected is unknown. The mmCIF that failed carried one category, `_atom_site`, and no `entity`, `struct_asym`, or `chem_comp` block.

A full lane run does not meet this. `target-preparer` writes `<attempt>/<phase>/structures/<target_id>.pdb`, so every stage after it reads PDB whatever format you supplied. The failing dispatch came from a canary that called the generator directly and skipped that stage.

Expect to be holding mmCIF. RCSB returns mmCIF, and the complex copy this helper writes keeps the source suffix, so the `--target-accession` route above leaves a `.cif` on disk. Run it through `target-preparer`, or convert it yourself, before any dispatch that bypasses the lane.

## When you already know the residues

Use `--surface-residues` when literature or mutagenesis already fixes the site. Entries are `CHAIN:NUMBER` or `CHAIN:START-END`, separated by commas, or a path to a file holding them one per line with `#` comments.

```python
subprocess.run([
    sys.executable, "-m", "claude_binder.make_target_inputs",
    "--structure", "target-structure.cif",
    "--chain", "A",
    "--surface-residues", "A:56,A:58,A:66,A:113,A:115,A:123-126",
    "--design-residues", "A:54-58,A:113-126",
    "--contact-cutoff-angstrom", "5.0",
    "--out", "pdl1.residue-map.json",
    "--site-out", "pdl1.site.json",
], check=True)
```

Every entry is checked against the chain, and an entry naming a residue the structure does not carry is refused. This route writes `site.mode` as `explicit-residues` unless `--site-mode` names another registered mode. The mode is then a label, because the residues still came from you.

Omitting `--design-residues` or `--contact-cutoff-angstrom` writes a `__REQUIRED__` placeholder in its place and prints a `TODO` line. `check` rejects the campaign until a person replaces it.

## When there is no partner structure

The helper never picks a surface from the target alone. Run it with a structure and no site and it stops:

```
make_target_inputs: no site was supplied. Pass --surface-residues with target
residues, or pass --target-accession and --partner to derive them from a
deposited complex.
```

Three things you can do instead.

- **Name the residues yourself** from literature, mutagenesis, or a homologous complex, and use the explicit route above.
- **Run unconstrained discovery.** Set `targets[].site.discovery` with `enabled` true in the campaign, and the lane sets `site.mode` to `unconstrained-discovery` and stops requiring `reference_contact_residues` and `design_residues`. Screen first, then group the candidates by the surface they actually contacted with `discover-contacts`. See [campaign fields](campaign-fields.md) for the field and [running a campaign](running-a-campaign.md#run-specialized-commands) for the command. The helper does not write this block, so add it by hand.
- **Generate the residue map alone.** A discovery target still needs `site.residue_map_path`. Use `--map-only` with `--structure`, `--chain`, and `--out`. It writes the chain-wide identity map without asking for a placeholder site or writing a site artifact:

  ```python
  subprocess.run([
      sys.executable, "-m", "claude_binder.make_target_inputs",
      "--structure", "target-structure.cif",
      "--chain", "A",
      "--map-only",
      "--out", "target.residue-map.json",
  ], check=True)
  ```

  `--map-only` refuses site and partner options so a partially specified site is not silently discarded.

`reference-pose-contacts` and `spatial-pocket` are registered mode names. The helper derives no residues for either, so the residues under those labels are still yours to supply.

## A target chain that is not protein

`chains_present` in the helper's output lists every chain in the structure, including nucleic acid, ligand, and cofactor chains. The helper maps the one protein chain you name with `--chain` and computes contacts against the partner chains you name. It does not classify the chemistry of the remaining chains, and it carries none of them into the residue map or the site block.

Decide whether a non-protein chain belongs in the target before you fold anything. Keep it when the site you are designing against exists only in its presence. A conformation held by a bound nucleic acid or cofactor presents a different surface without it, so a binder designed against the stripped coordinates is designed against a different target.

Declare a kept chain in the campaign. `targets[].entities[].type` registers `protein`, `rna`, `dna`, `ligand`, and `cofactor`. See [campaign fields](campaign-fields.md) for the entity block.

A kept nucleic acid chain bears on the co-folder. The [tool catalogue](tool-catalogue.md) records a measured case where ESMFold2 misplaces RNA coordinates in both model sizes, scoring DockQ 0.009 and 0.015 where Boltz-2 scored 0.415 on the same case. Read that as a warning about accuracy at a protein-RNA interface, not as a reason to withhold the chain from an arm.

Supply the kept chain to every ranking arm. The release requires it and says never to waive a dossier-flagged cofactor on the assumption that an arm is protein-only. Read that clause literally, because the mistake it names is easy to make. ESMFold2 takes non-protein context by Chemical Component Dictionary code in both model sizes, and the measured protein-RNA case above exists only because it accepted RNA and placed it badly. Its single-sequence description refers to the missing MSA encoder, not to a missing nucleic-acid channel, and one is not evidence of the other.

The floor sits under that. When the dossier flags a cofactor as fold-required or interface-required, at least one ranking arm has to represent it. An all-cofactor-blind ranking on such a target is a logged deviation rather than a free choice, recorded per arm alongside the cofactors present and the number of target chains folded. The validation gate decides which arms survive on a target, because its fold-recapitulation condition is what establishes whether an arm represents that target at all. When an arm turns out to be unavailable or unvalidated there, the release names its own substitution ladder: one independent-lineage co-folder per missing arm, in the order AlphaFold-Multimer-v3, then AlphaFold3 code with OpenFold3 weights, then Chai-1, then Boltz-2 or Boltz-1, all with target-chain MSAs, recording which arm each stood in for.

Taking that ladder follows the release rather than departing from it. The package's `baseline_fidelity` flag still will not assert, because it tracks the exact published adapter IDs, and a withheld claim stops no stage and disables no tool. Record the stand-in under [reproduce or substitute](the-nine-decisions.md#2-reproduce-or-substitute). [Published targets](published-targets.md) carries the construct rules, and [reproduction readiness](reproduction-readiness.md) prints what a profile still binds.

## Residue numbers belong to one structure

A residue label means nothing without the file it was read from. Site residues, design residues, and the residue map are all in the numbering of `targets[].structure_path`. Copying residue numbers between two structures of the same protein puts you on the wrong surface while every file validates.

The shipped `campaign.example.json` is a non-evaluative wiring fixture. Its target and its provider-derived pose-recovery control point to the same RFdiffusion3 canary output. Its site therefore uses `reference-pose-contacts`, not `reference-partner-contacts`, and neither the target nor the control is evidence that a candidate binds. Use a freshly prepared experimental target and an independent control panel for a real campaign.

The fixture's `source_id` is the UniProt accession `Q9NZQ7`, and its site residues are `B:37` through `B:106`. Those numbers are not UniProt numbers. Its structure model's chain B is the PD-L1 construct renumbered from 1, so the frames differ by a constant.

Derive that constant from the files rather than assuming it. The design model's chain B is 115 residues numbered 1 to 115. Chain A of PDB 4ZQK is the same 115-residue sequence, numbered 18 to 132, and 4ZQK records that range as Q9NZQ7 18 to 132. The sequences are identical, so `B:N` in the example is Q9NZQ7 residue `N + 17`. Every one of the 115 residue identities agrees at `+17` and disagrees at any other offset.

Apply it and the example's 16 pose-derived residues become `I54 Y56 E58 D61 K62 N63 Q66 V68 H69 H78 R113 M115 S117 A121 D122 Y123` in Q9NZQ7 numbering. Eleven overlap the 22 independent PD-1 contact residues derived above from 4ZQK and five do not. The partial overlap does not turn the generated pose into a partner-derived epitope.

Read a site block together with the structure it was written against. A number copied out of one frame and into another is off by the whole offset, and nothing downstream will catch it.

## The residue map is the identity

`source_to_cleaned` translates a label in the structure file into the label the same residue carries in the structure handed to the predictors. The helper writes the identity over every residue of the chain, which is correct when `targets[].structure_path` is already the numbering the predictors receive.

The offset to a reference numbering belongs outside this map. Both ends of this map are inside one campaign, and `compute_site_metrics` applies it to the site labels before matching them against the predicted pose. Writing the `+17` above into `campaign.example.residue-map.json` would produce site labels the predicted structure does not carry, and every site metric would match nothing. The example's all-identity map is correct, and the frame it is in is documentation.

The helper supplies no mapping source for a cleaned or renumbered target structure. Write the residue map yourself when residue numbers differ between the deposited and the folded structure.

Source: [`make_target_inputs.py`](../claude_binder/make_target_inputs.py) defines the arguments and the generated files. [`partner_site.py`](../claude_binder/partner_site.py) resolves the complex and computes the contacts. [`target_prep_adapter.py`](../claude_binder/adapters/target_prep_adapter.py) normalizes the structure the run's later stages read. [`test_make_target_inputs_cli.py`](evidence/tests/test_make_target_inputs_cli.py) and [`test_partner_site_resolution.py`](evidence/tests/test_partner_site_resolution.py) test both routes.
