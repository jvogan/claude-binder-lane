# TSLP demo campaign

Design binders against the TSLPR-engaging face of human TSLP.

Every number in this directory was computed from the deposited coordinates it ships, at a
5.0 angstrom heavy-atom cutoff. Nothing here was recalled or estimated.

## Why this target

TSLP carries three separate reference interfaces on one 147-residue chain, and two of them barely
touch. That gives a positive control and a genuine alternate-site negative control from real
depositions, which is what `controls` requires.

The design site is the TSLPR face. Tezepelumab, the antibody in the approved drug Tezspire, covers
8 of the 25 residues on that face and none of the 17 residues on the IL-7Ralpha face. So the demo
designs against the face an approved drug already blocks.

## Structures

All four files are chain subsets written from RCSB mmCIF entries. Hydrogens and secondary altlocs
are dropped, because the campaign declares `atom_selection: "heavy-atoms"` and both source entries
carry riding hydrogens from refinement. Keeping them inflates every contact count by 4 to 5
residues.

| File | Source | Chains | Use |
| --- | --- | --- | --- |
| `structures/tslp-target.pdb` | 5J11 | A | design target, 114 observed residues, 28-159 |
| `structures/tslp-tslpr.pdb` | 5J11 | A, C | positive control, TSLP with TSLPR |
| `structures/tslp-tezepelumab.pdb` | 5J13 | A, C | positive control, TSLP with the tezepelumab heavy chain |
| `structures/tslp-il7ra.pdb` | 5J11 | A, B | negative control, TSLP with IL-7Ralpha |

5J11 is TSLP with TSLPR and IL-7Ralpha, X-ray, 2.56 angstrom, released 2017-04-05.
5J13 is TSLP with the tezepelumab Fab, X-ray, 2.298 angstrom, released 2017-04-05.
Chain A residue numbering is identical between the two entries at all 108 shared positions.

## Measured interfaces

| Interface | Residues | Positions on TSLP |
| --- | --- | --- |
| TSLPR (CRLF2), the design site | 25 | 43-44, 48, 60-69, 142, 145-147, 149-150, 152-153, 156-159 |
| IL-7Ralpha | 17 | 40-42, 44-47, 49-50, 53, 97, 100-101, 104-105, 108-109 |
| Tezepelumab heavy chain | 15 | 64-72, 74-75, 78, 150-151, 153 |

Overlaps by Jaccard index: TSLPR against IL-7Ralpha is 0.02, one shared residue. Tezepelumab
against TSLPR is 0.25, eight shared residues. Tezepelumab against IL-7Ralpha is 0.00.

The tezepelumab light chain makes zero contacts with TSLP at this cutoff, so the heavy chain alone
carries the whole epitope and fits the single-chain `binder_chain` field.

No non-water heteroatom is within 5 angstroms of either receptor site, so the 28 NAG glycans in
5J11 do not touch the design site.

## Target identity

UniProt Q969D9, 159 residues, confirmed against the UniProt REST API and against the RCSB entity
mapping for 5J11.

## What is not resolved

`gates/tslp.json` does not exist. The control gate thresholds in `campaign.json` are the values
carried by `campaign.example.json`, not TSLP measurements. Producing real thresholds requires a
control-calibration run, which is paid compute. Treat them as placeholders until that run exists.

`scoring.counter_screen.off_target_paralog_id` carries a TODO. This demo has no paralog antitarget.
The alternate-site negative control carries the specificity test instead.
