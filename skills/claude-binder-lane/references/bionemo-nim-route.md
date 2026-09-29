# NVIDIA BioNeMo NIM route

NVIDIA BioNeMo NIM gives Claude Science another execution route for binder
generation, sequence design, MSA search, and co-folding. Treat NIM as the
service surface for a selected model. Record the model, endpoint, deployment,
inputs, and outputs separately.

The NVIDIA tutorial published on 2026-08-31 runs MSA Search, OpenFold3, and
Boltz-2 NIM endpoints through Claude Science. NVIDIA's BioNeMo Agent Toolkit
also documents a binder workflow that composes RFdiffusion, ProteinMPNN,
Boltz-2 or OpenFold3, and optional MSA Search. These records establish a route
shape. They do not establish that an endpoint is connected in a particular
Claude Science session.

The toolkit catalogue labels its hosted OpenFold3 and MSA-to-OpenFold3 agent
evaluations as pending after timeouts. That label describes NVIDIA's automated
skill evaluation, not every OpenFold3 endpoint. Keep the route visible, then
use a health check and N=1 artifact canary for the endpoint the scientist
selects.

## Choose the execution surface

| Surface | What Claude Science uses | What you record |
| --- | --- | --- |
| Hosted NIM | A provider-managed HTTPS endpoint | Model and service revision, endpoint identity, request schema, price source, egress decision, response fields, and provider job identifier |
| Local NIM | An NVIDIA container on a GPU machine | Image digest, model profile, weights, database versions, GPU, mounts, readiness result, and container logs |
| Remote NIM | A local NIM container reached through SSH, HPC, Modal, or another cloud | Every local-NIM field plus provider, environment identity, authorization result, rate evidence, job identifier, and cleanup result |
| Binder adapter | A model-specific package contract | Every route field plus argv or JSON templates, parser result, receipts, spend reservation, resume behavior, and artifact hashes |

Discover the live Claude Science skill and endpoint inventory before choosing
one of these surfaces. A visible BioNeMo family or imported skill is planning
evidence. A registered endpoint with a successful free health check is route
evidence where the surface offers one. Measured in a Claude Science run on
2026-08-31: a hosted NIM endpoint exposes no start, stop, or live call,
so a hosted route has no free health check and its first evidence is the N=1
call. An N=1 result with parsed artifacts is scientific contract evidence.

Binder does not need to own the endpoint before a scientist can use it.
OpenFold3 remains unbound in `catalog.json`, so Binder cannot place it directly
in an executable plan. Claude Science can still call a visible OpenFold3 NIM
runbook, save its outputs, and return the hashed artifacts through a manual
handoff. Add a Binder adapter only when the campaign needs automated dispatch,
resume, and receipt handling for that exact route.

## Atomic tools relevant to a binder campaign

The BioNeMo Agent Toolkit revision recorded in the sources lists these NIM
skills:

| Campaign role | NIM skill | Primary handoff |
| --- | --- | --- |
| Backbone generation | `rfdiffusion-nim` | Target structure, contigs, and hotspot residues to binder backbone PDBs |
| Sequence design | `proteinmpnn-nim` | Backbone PDB and designed-chain selection to designed sequences and scores |
| Evolutionary context | `msa-search-nim` | Protein sequences and search policy to A3M alignments |
| Independent co-folding | `openfold3-nim` | Molecules and explicit MSA fields to mmCIF and confidence outputs |
| Independent co-folding | `boltz2-nim` | Molecules and per-chain MSAs to mmCIF and confidence outputs. PAE availability depends on the deployment; see the measured note below |

The toolkit's workflow is a useful composition reference, not a fixed Binder
stack. A campaign can use one NIM stage and choose local, hosted, or self-hosted
routes for the others, but not from a single Claude Science host. Measured in a
Claude Science run on 2026-08-31: a host's NIM mode is exclusive, so
a host in the platform's URL (remote) mode refuses local container registrations
and a host in local mode refuses URL ones. That platform sense of "remote" is not
the Remote NIM row above. Mixing modes takes one host per mode. Preserve each
handoff contract and each route's cost and authorization record.

## Choose the MSA topology from the scientific question

| Question | MSA input |
| --- | --- |
| De novo binder against a target | Use a target-chain unpaired MSA when the selected model benefits from it. Keep the designed binder as a single sequence. Do not fabricate binder homologs or a paired binder-target MSA. |
| Natural protein complex | Use per-chain unpaired MSAs. Add a species-paired MSA when the method supports it and the pairing policy is recorded. |
| MSA ablation | Run the same sequences and model with and without the MSA. Keep sampling settings constant and label both conditions. |
| MSA unavailable | Stop if the approved protocol requires it. Continue without an MSA only when the scientist selected that condition. |

The NVIDIA tutorial's Seh1 example is a natural-protein interaction study. Its
with-MSA interface scores exceeded its no-MSA scores for OpenFold3 and Boltz-2.
That result supports the tutorial's natural complex and does not make a paired
MSA appropriate for a de novo binder.

OpenFold3 and Boltz-2 use different request shapes. OpenFold3 accepts a
per-chain `msa` and a separate `paired_msa` for a complex. The Boltz-2 NIM
accepts a per-chain MSA and performs pairing internally; its service also
exposes an optional `concatenate_msas` setting. Record the native request. Do
not translate one model's MSA field into another model's field by name alone.

## Preserve native scores before comparison

Save every confidence field and its scale. In the NVIDIA tutorial, OpenFold3
returned a scalar `iptm_score`, while Boltz-2 returned scalar, per-chain,
pairwise, PAE, and PDE arrays. The tutorial also reports pLDDT on a 0 to 100
scale for OpenFold3 and a 0 to 1 scale for Boltz-2.

**Measured 2026-09-11, and it does not match the tutorial.** On the registered
remote endpoints, neither model returns a PAE matrix. Boltz-2 declares
top-level `pae` and `pde` keys and both are `None` on every call;
`write_full_pae` is accepted without a 422 and changes nothing; sixteen other
candidate field names were rejected as extra inputs. OpenFold3 returns scalars
only, under `outputs[0]["structures_with_scores"][0]`: `iptm_score`,
`ptm_score`, `complex_plddt_score`, `complex_pde_score`, `confidence_score`.
The scale difference above is confirmed -- OpenFold3 reported 89.90 where
Boltz-2 reported 0.9577 on the same complex.

Those recorded deployments cannot supply ipSAE from these responses. ipSAE
requires the PAE matrix, which cannot be recovered from the returned mmCIF.
Choose a deployment that returns a usable matrix when the planned score needs
it. Hosted and self-hosted routes can qualify; verify the selected response
before scaling. One recorded option is self-hosted Boltz with full PAE output.
The tutorial and these observations describe different deployments. Preserve
the service revision and response evidence, and read
[co-fold scoring hazards](cofold-scoring-hazards.md) for the scoring contract.

Two further measured notes on these endpoints. The POST path is
`BASE_URL + "/predict"`; a POST to `BASE_URL` itself returns a bare
`404 page not found`, which reads like a dead service. Per-chain MSA nesting is
`msa[database][format]`, matching what `msa-search` returns, so a response's
`alignments` entry passes straight through; inverting the two levels produces an
error naming the format value, which misdirects. Boltz-2 also runs
single-sequence silently when `msa` is omitted -- measured ipTM 0.6105 without
an alignment against 0.9293 with one on the same design, and a median lift of
0.2285 across eight designs -- so assert the alignment arrived rather than
trusting a 200.

Compare conditions within one model before comparing models. A shared field
name does not guarantee shared calibration. Normalize a value only when the
model documentation defines the scale conversion, and retain the native value
beside the normalized value.

For each call, preserve:

- input sequence accessions, chain IDs, lengths, and checksums;
- target structure and residue-map digests;
- unpaired and paired A3M files with database, database version, search type,
  E-value, depth cap, and pairing policy;
- request and response documents;
- endpoint family, service revision, container image digest, model profile,
  and weights revision when the surface exposes them;
- every returned structure sample, confidence field, PAE or PDE output, runtime
  field, warning, and error;
- provider job identifiers, authorization result, estimated charge, settled
  charge when available, and cleanup result.

An empty runtime field stays empty. Do not estimate a model-reported field from
another output.

## Size local deployment from the selected profile

The NVIDIA tutorial's local setup uses an L40S or H100 and about 700 GB of
storage. Its UniRef30-only MSA profile accounts for about 490 GB, and its
OpenFold3 and Boltz-2 containers account for another 30 to 40 GB. NVIDIA's MSA
Search skill states that the full database profile is about 1.2 to 1.4 TB.

These figures describe the cited images and database profiles. Read the
selected image's model profiles before provisioning storage. A hosted endpoint
removes the local database requirement but introduces provider pricing, data
egress, retention, and service limits.

## Return NIM results to Binder

1. Record the selected NIM skill, endpoint surface, model, MSA policy, budget,
   and artifact destination in the campaign decision record.
2. Run the endpoint through its Claude Science runbook. Use a free health check
   first where the surface offers one, and an N=1 artifact canary when the route
   has no recorded result. A hosted endpoint has no free health check, so budget
   for the canary call itself.
3. Save the native request, response, structures, alignments, receipts, and
   hashes outside an expiring session workspace.
4. Import the result at the next Binder boundary. Use a manual handoff for a
   one-off stage. Use the supplied-candidate manifest when the result carries
   the candidate structure, design pose, sequence, target identity,
   residue-map identity, lineage, and hashes that schema requires.
5. Add a package adapter only when repeated campaigns need Binder to dispatch
   and resume the same endpoint contract.

## Sources

- [NVIDIA tutorial: BioNeMo NIM protein structure prediction in Claude Science](https://developer.nvidia.com/blog/run-nvidia-bionemo-nim-microservices-for-protein-structure-prediction-in-claude-science/), published 2026-08-31.
- [BioNeMo Agent Toolkit catalogue](https://github.com/NVIDIA-BioNeMo/bionemo-agent-toolkit/tree/0e67a612e4045f007e38fa77adc8f3ebfc5616b6), revision `0e67a612e4045f007e38fa77adc8f3ebfc5616b6`.
- [BioNeMo protein-binder workflow](https://github.com/NVIDIA-BioNeMo/bionemo-agent-toolkit/blob/0e67a612e4045f007e38fa77adc8f3ebfc5616b6/workflows/generative-protein-binder-design/protein-binder-design/SKILL.md).
- [BioNeMo MSA Search NIM skill](https://github.com/NVIDIA-BioNeMo/bionemo-agent-toolkit/blob/0e67a612e4045f007e38fa77adc8f3ebfc5616b6/nim-skills/msa-search-nim/SKILL.md).
