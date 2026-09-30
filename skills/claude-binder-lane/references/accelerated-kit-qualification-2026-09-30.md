# Native Modal kit qualification, 2026-09-30

This record covers actual H100 inference on public upstream inputs using
Anthropic's release commit
`f4f62fa6592ae4938d49b1757bea0cfeff9f468e`. Read the
[integration guide](accelerated-kit-integration.md) for account-local setup and
workflow handoffs. These are engineering qualification runs with small valid
counts and full scientific inference settings. They do not establish campaign
quality, production throughput, or reproduction of the published binder study.

| Kit | Tested modes | Actual output and scope |
| --- | --- | --- |
| ProteinMPNN | Vanilla and soluble: `exact`, `off` | Two full-backbone public monomers; one designed sequence per monomer, parsed scores and probabilities. [Evidence](evidence/accelerated-kits-2026-09-30/proteinmpnn.json). |
| Genie3 | `exact`, `off`, `fast` | One PD-L1 design per mode with the full 100-step sampler; parsed chain roles and C-alpha binder geometry. [Evidence](evidence/accelerated-kits-2026-09-30/genie3.json). |
| PXDesign | `exact`, `off`, `fast`, `big` | One full-backbone PD-L1 complex per mode with the full 400-step sampler; target 116 residues and binder 75 residues. [Evidence](evidence/accelerated-kits-2026-09-30/pxdesign.json). |
| RFdiffusion3 | `exact`, `off`, `fast` | One 95-residue binder per mode; full backbone, preserved 115-residue target sequence and unchanged public contig/hotspots. [Evidence](evidence/accelerated-kits-2026-09-30/rfdiffusion3.json). |
| BoltzGen | `exact`, `off`, `fast`, `big` | Two designs per mode through generation, sequence design, complex and binder-only refolding, analysis, and filtering. [Evidence](evidence/accelerated-kits-2026-09-30/boltzgen.json). |
| Complexa | `exact`, `off`, `fast`, `big` | Two full-backbone PD-L1 replicas per mode with default best-of-n search and actual AF2 rewards. [Evidence](evidence/accelerated-kits-2026-09-30/complexa.json). |

ProteinMPNN's monomer exact/off comparisons produced identical FASTA bytes and
identical values in every stored NPZ array. Its startup probes passed and its
manifests recorded no partial execution or fallback. Upstream reported
`sort_by_length` as unobserved; the small workload does not demonstrate that
lever's benefit.

The real [PXDesign/RFdiffusion3 handoff](evidence/accelerated-kits-2026-09-30/proteinmpnn-handoff.json)
then designed both generated complexes with vanilla and soluble weights in
Exact and Off. All eight designed outputs passed; Exact/Off FASTA bytes and
every saved NPZ array value matched. Saved sampled residue indices, chain
orders and loss masks independently confirmed 924 unchanged fixed-target
residue observations and enabled binder design. Format conversion retained
every atom identity and recorded at most 0.0005 angstrom coordinate rounding.
PXDesign's unknown binder placeholders are explicitly represented as `UNK`/`X`;
no missing atoms or sequences were reconstructed.

Genie3's exact/off PDBs and canonical atom coordinates were identical. All
requested accelerated levers had execution evidence. A second qualification
ran the final shipped recipe directly: its executed and current recipe hashes
match, and all three modes' outputs match their corresponding first runs in
both bytes and canonical coordinates. Its Fast run included initial
compilation and took longer than Exact at this count; that observation
supports no throughput speedup claim.

Genie3's binder is C-alpha-only. Full-backbone ProteinMPNN Exact cannot consume
it directly. Use a compatible C-alpha checkpoint through its documented stock
or native route, or qualify a distinct reconstruction stage. Retain the target
representation and residue mapping when selecting the next consumer.

RFdiffusion3's Exact/Off atom-site records were identical with deterministic
execution. Gzip container hashes differed, while the companion JSON hashes
were identical. Its requested kernels and graph counters were observed; Fast
also recorded compilation and served gather calls without fallback.

PXDesign's shipped recipe applies one hash-gated, disclosed kit-driver patch:
it removes an unused accelerator import from the stock route. The original
driver failed the strict after-call isolation check. The repaired driver
passes that check, with empty kit and lever module lists. Every mode's input,
scientific parameters, CIF bytes and canonical coordinates match its
corresponding original-driver run. Exact/Off outputs also match each other.
The patch, before/after driver hashes and comparison evidence are retained.

BoltzGen retained 500 generation steps, 200 steps for the other GPU stages,
three recycles, and its default selection budget of 30. Each mode produced
two 66-residue binders and a preserved 115-residue target sequence. Refolding
can change target geometry. Exact/Off structures
and coordinate archives matched; two `design_iiptm` arrays differed among 112
compared array keys. Their maximum absolute difference was approximately
0.00315. All other scores and filter flags matched. Whole-pipeline bitwise
equality is therefore unproven despite the matching structures. The bundled
confidence implementation contains a repeated-index CUDA overwrite that is a
possible explanation, inferred from source rather than established by a
controlled test. [PyTorch documents nondeterministic behavior for duplicate
indices](https://docs.pytorch.org/docs/main/generated/torch.Tensor.scatter_.html).

Exact and Off each returned one filter-passing candidate; Fast and Big each
returned none on this small input. Upstream retains failed candidates in its
ranked output. Rank presence does not assert filter success or binding quality.

Complexa preserved its 400-step sampler, two best-of-n replicas, AF2 Multimer,
three recycles, initial guess and default interface-PAE reward. Each mode
produced two target-A115/binder-B80 complexes and two finite AF2 reward rows
on JAX's actual GPU device. The upstream default evaluates AF2 model 0 once
per reward; five model parameter sets are available. No model-selection
override or reward removal was introduced. Exact and Big PDB bytes and all
non-path reward CSV values matched Off. Fast differed as expected. Every
requested lever served; Fast and Big also recorded 24 upstream guarded-stock
calls. Separate filtering, evaluation and analysis workflows were not run.

These runs use the native Modal SDK and the shipped recipe interfaces.
Claude Science runtime 0.1.54 supports the custom recipe contract, but its
session submission and installed-skill execution were not exercised. No full
Binder graph adapter or scientific outcome is implied by a native inference
pass.

Raw provider receipts and artifacts stay outside the public package. The linked
JSON summaries retain public input identities, artifact hashes, parsed counts,
parameters, activation, and limitations. Private account names, resource IDs,
host paths, and credentials are omitted. Cost admission bounds and provider
settlement are tracked separately; no settled task charge is asserted here.
