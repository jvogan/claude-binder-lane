# Tool bring-up

Read this after discovery and before qualifying a selected route.

Use the exact route record as the source of runtime truth. Where the tool is one the release ran, its own run record is the closest thing to a reference configuration: `data/docs/LOOKUP_TABLES.md` in the released dataset carries, per predictor, the invocation, the pinned software version, the checkpoint and where to get it, the MSA mode, the seed count, and the cofactor and sgRNA handling. Read it alongside the pins below, which were read from the publishers rather than from that table. [Published targets](published-targets.md) lists which predictors ran and on which construct. A visible skill is a
runbook. A reusable image, endpoint, API, local install, or compute environment
can qualify when its identity and contract evidence match the campaign. Building
every tool from source is unnecessary.

## Contents

- [What bring-up costs and who does it](#what-bring-up-costs-and-who-does-it)
- [What PASS means](#what-pass-means)
- [Three traps](#three-traps)
- [When a model becomes required](#when-a-model-becomes-required)
- [Install route, the published three ranking modes](#install-route-the-published-three-ranking-modes)
- [Install route, the three scoring tools](#install-route-the-three-scoring-tools)
- [Install route, the LigandMPNN sequence designer](#install-route-the-ligandmpnn-sequence-designer)
- [Building the image this skill ships](#building-the-image-this-skill-ships)

## What bring-up costs and who does it

You do it. This page carries the pins, the weight repositories, the Python
constraints, the ready-made environments and the traps, so bring-up is work to
carry out rather than work to hand back. Do not ask the scientist to install a
runtime, resolve a pin, or build an image.

Four things are the scientist's, and every one of them is a decision rather than
a task: spending beyond a stated ceiling, any change of provider, hardware or
scale that changes the bill, the target and site, and the controls and
thresholds. Anything only they hold comes from them too, such as an account, a
credential, or an executed written agreement. `SKILL.md` sets out the full
division under what you may settle without asking.

Bring-up is route-dependent. Inventory and account checks are usually free.
Local import or `--help` checks are usually free. Use the smallest N=1 canary
that proves the selected input, output, parser, and artifact contract. Parallel
workers are optional. Price paid canaries before launch and charge them to the
campaign ceiling.


## What PASS means

Promotion is evidence-specific:

1. **Documented** means a current runbook or official service contract is visible.
2. **Configured** means the selected executable, image, endpoint, API account, or
   compute environment has a recorded identity.
3. **Reachable** means a free route-specific probe succeeded.
4. **Contract-verified** means the smallest meaningful canary produced the expected
   entity, chain, residue, file, count, and parser contract with hashes.
5. **Campaign-qualified** means the exact route also has the required model and
   weight pins, licence and data-handling fit, price basis, controls, downstream
   handoff, receipts, and cleanup evidence for the campaign's claim.

Only the selected route needs promotion. A contract-verified alternative can
remain visible without blocking another route. Reuse a prior image or service
qualification when its immutable identity, contract revision, scientific scope,
and relevant provider facts still match. Run a fresh canary when any of those
facts changed or cannot be established.

## Three traps

**Backbone format compatibility**. Several structure-design models emit
non-standard backbone files: C-alpha-only PDBs, non-standard binder residue names,
swapped binder and target chain order. Validate the complete sequence-design adapter pipeline for each model.

**Protenix compilation**. The first Protenix v2 call compiles a CUDA kernel. This requires four to six minutes on an H100.
Point the torch extensions cache at a persistent volume path shared across containers.
Give the first bring-up call at least a fifteen-minute timeout.

**Package identity**. `pip install esm` gets the right package. ESMFold2 entered the PyPI `esm`
package at version 3.4.0 on 2026-08-27, and 3.4.1.post1 ships `esm/models/esmfold2/` under a Chan
Zuckerberg Biohub MIT copyright. Installing from `github.com/Biohub/esm` gets the same code at the
same version. Read 2026-09-16 from the published wheel and from the project's `pyproject.toml`.

## When a model becomes required

No model gates every campaign. A model becomes required when the scientist
selects it, when a chosen published-baseline claim names it, or when the campaign
declares a control or independent-lineage rule that only that route satisfies.

If a required route cannot qualify within the scientist's budget or time limit,
choose another route and disclose the substitution, narrow the claim, or end that
attempt. Other tools and campaign shapes remain available. The scientist chooses
the canary budget and retry limit before paid work.

## Install route, the published three ranking modes

Every route below was read off the tool's own repository or off a bundled environment
file on this install. Read each one again at kickoff, because a pin drifts.

### ESMFold2-Full and ESMFold2-Fast

Both arms come from one package and one image.

- **Code**: `esm` 3.4.1.post1, from PyPI or from `https://github.com/Biohub/esm` at a pinned
  commit, on **Python 3.12 or later**. The package declares `requires-python = ">=3.12"`, so it
  cannot share an image with a Python 3.11 stack. It also pins `torch>=2.11.0,<2.12.0` and
  `transformers>=4.57.6,<5.0.0`.
- **Weights**: `biohub/ESMFold2` and `biohub/ESMFold2-Fast` on Hugging Face. Ungated,
  MIT, about 6.5 billion parameters each across six safetensors shards. ESMFold2 also depends on
  ESMC-6B embeddings, so `biohub/ESMC-6B` comes down with them. Revisions read 2026-09-16:
  `69869f73`, `45fe8656`, and `af1602ba`.
- **Licence**: MIT for code and weights, with a separate Biohub Acceptable Use Policy
  layered on top.
- **Ready-made image**: the compute skill ships an `esmfold2_gpu` environment carrying
  this stack, with the three weight repositories snapshot-downloaded at pinned commits
  into a cache volume and `HF_HUB_OFFLINE=1` set. Passing `env: 'esmfold2_gpu'` in
  `provider_params` applies that environment's own egress declaration. Treat it as a
  starting point and still run your own PASS check against it.

**The weight repositories changed on 2026-09-14.** `tokenizer_config.json` moved
`extra_special_tokens` from `[]` to `{}`, which the upstream commit titles a fix for transformers
4.x. An earlier commit that day removed the unbundled single-file checkpoint. A cache volume
hydrated before that date holds the old tokenizer config and a checkpoint the current config does
not reference. Re-hydrate it rather than reuse it.

Two facts about that image matter. `flash-attn` is deliberately absent, because the
trunk speedup comes from a vendored fused backend and flash-attn added no measured trunk
speedup. The transformers-path model class cannot be imported without a live CUDA driver,
because autotune runs at import time, so a CPU sandbox cannot smoke-import it.

### Protenix v2

One published baseline names Protenix v2. The package ships its adapter and a
full-ensemble profile route, but it does not bundle a qualified CUDA image,
immutable source and weight revisions, settled weights terms, or a completed
runtime validation. Those missing operator facts block execution; the local
kernel architecture and access to any one hosted application do not block a
self-hosted route.

- **Code**: `https://github.com/bytedance/Protenix`.
- **Weights**: pinned Protenix v2 source maps model name `protenix-v2` to
  `https://protenix.tos-cn-beijing.volces.com/checkpoint/protenix-v2.pt`.
  Upstream publishes no object digest, and the checkpoint URL answers HTTP 403,
  so the official bytes remain unqualified. A third-party mirror is a separate
  route unless its bytes match an upstream digest and its terms are known.
- **Licence**: Apache License, Version 2.0, for both code and model parameters. The
  README states the extension to parameters in one sentence, and the repository LICENSE
  covers the code. Read 2026-08-21.
- **Image**: build from the pinned recipe or reuse an immutable matching image.
  Record its digest, environment identity, model and weight hashes, and a
  route-specific qualification receipt.

The pinned source, base-image family, dependency-lock design, hydration contract,
network phases, and required qualification receipts are specified in
[Protenix v2 bring-up](protenix-v2-bringup.md). The route remains fail-closed until
the official checkpoint can be retrieved and its reviewed SHA-256 manifest committed.

## Install route, the three scoring tools

### ipSAE

- **Source**: `https://github.com/DunbrackLab/IPSAE`.
- **Install**: download the single script `ipsae.py`. Its only stated dependency is
  `numpy`.
- **Invocation**: `python ipsae.py <scores_file> <structure_file> <pae_cutoff>
  <dist_cutoff>`. The scores file is the predictor's JSON, or a PAE `.npz` for
  predictors that write one, and the structure file is the matching PDB or CIF.
- **Which column**: the plain `ipSAE` column is the **d0res** variant, which is the one
  the released re-score used at a PAE cutoff of 10 angstroms. `ipSAE_d0chn` and
  `ipSAE_d0dom` are different normalizations and are not interchangeable with it.
- **What it prints**: one asymmetric row per direction plus a `max` row per chain pair.
  It prints no minimum row. `ipSAE_min` is the protocol's metric, so you compute the
  minimum over the two asymmetric rows yourself.

### DockQ

- **Source**: `https://github.com/bjornwallner/DockQ`.
- **Install**: `pip install DockQ` puts a `DockQ` binary on the path. For the latest
  commit, clone the repository and `pip install .` inside it.
- **Invocation**: `DockQ <model> <native>`. With more than one interface it reports a
  result per interface, computed to maximise the average across them.
- **Version**: the release pins **DockQ v2**. Confirm the installed version before you
  score anything, because v1 and v2 report different numbers.

### MMseqs2

MMseqs2 backs the novelty gate's sequence search against UniRef90 and the known-binder
corpus.

The pinned MMseqs2 release, UniRef90 `2026_02` bundle format, offline checker,
database-build argv, no-egress search contract, and capacity work still required are
recorded in [MMseqs2 + UniRef90 local route](mmseqs2-uniref90-local-route.md).
Do not promote the route until its full-database memory, disk, and wall-time canary is retained.

## Install route, the LigandMPNN sequence designer

`ligandmpnn-designer.template.json` puts LigandMPNN on the sequence-design decision in place
of ProteinMPNN. The adapter runs the publisher's own `run.py`, so bring-up is a checkout, a
pinned NumPy, and two checkpoint files.

Qualification is free. A CPU run with CUDA unavailable took 2.07 seconds for 93 designed
residues at one design, and 3.16 seconds for 177 residues at two designs, both measured on
2026-09-10 with torch 2.12.0.

1. Clone the publisher's repository and check out the pin this package was read against.

   ```sh
   git clone https://github.com/dauparas/LigandMPNN
   cd LigandMPNN && git checkout 26ec57ac976ade5379920dbd43c7f97a91cf82de
   ```

2. Install from `requirements.txt`, which pins `numpy==1.23.5`.

   ```sh
   pip install -r requirements.txt
   ```

   That pin matters. `run.py` imports `sc_utils`, which imports a vendored copy of
   openfold at module scope, and that copy reads `numpy.int`, removed in NumPy 1.24. A modern
   NumPy raises `AttributeError: module 'numpy' has no attribute 'int'` on every invocation.
   Installing the dependencies by name instead of from the file resolves a modern NumPy and
   reproduces that failure.

3. Fetch the checkpoints. The publisher serves them from `files.ipd.uw.edu/pub/ligandmpnn/`
   with no account, agreement, or click-through.

   ```sh
   bash get_model_params.sh "./model_params"
   ```

   The script downloads 15 files totalling 123 MB. `ligandmpnn-designer.template.json`
   selects `--model-type ligand_mpnn`, whose default checkpoint is
   `ligandmpnn_v_32_010_25.pt` at 10.5 MB, so one file covers the shipped profile. Each
   model type has its own checkpoint flag, so the adapter passes a file path rather than a
   weights directory.

4. Point the profile at the checkout with `--ligandmpnn-root`, or set `LIGANDMPNN_ROOT`.

Read [licence and commercial use](licence-and-commercial-use.md) before a commercial
campaign. The code and the model parameters are both MIT at the pin above.

### Two traps specific to this tool

**It exits 0 having designed nothing.** Name a chain LigandMPNN cannot design, a nucleic-acid
chain for example, and the run writes the FASTA, one backbone PDB per requested design, and a
native header reading `num_res=0`, while every returned record repeats the native sequence.
Counting outputs and reading the exit code both accept that run. The adapter parses the
residue count and refuses, naming the chain. Give `--design-chain` a protein chain of the
input structure.

**Its flags are not ProteinMPNN's.** Five that `proteinmpnn_designer.py` sends do not exist
here: `--num_seq_per_target`, `--sampling_temp`, `--path_to_model_weights`,
`--use_soluble_model`, and `--pdb_path_chains`. Use `--batch_size` with
`--number_of_batches`, `--temperature`, the per-model checkpoint flag, `--model_type
soluble_mpnn`, and `--chains_to_design`. This matters when you script `run.py` yourself; the
shipped adapter already builds the correct argv.

## Building the image this skill ships

`envs/binder_scoring_gpu.py` is the image recipe for Protenix v2 and the two scoring
tools. It is a file inside this skill, and the environment builder reads from the
workspace working directory, so copy it there first.

1. Copy the recipe out of the skill and into the workspace working directory with
   `host.skills.read('claude-binder-lane', 'envs/binder_scoring_gpu.py')`.

2. Before you build, check whether the environment already exists.
   `compute_details({provider: 'modal', mode: 'read'})` returns a per-workspace ledger of
   `### env:<name>@<spec_sha>` blocks, and a block whose `spec_sha` matches the current
   release is a reuse candidate. That ledger is shared across sessions. Verify
   its identity, access, and qualification receipt before selecting it.

3. In the `compute_provider` tool, build it:
   `r = build_env('binder_scoring_gpu', path='./binder_scoring_gpu.py', hydrate=True)`.
   Print `r['image']`, `r['spec_sha']` and `r['volumes']`, and save the record.

4. Read the recipe's outbound-host declaration, `META["egress_domains"]`, before the
   first run. Under an allowlist network policy a job reaches only the merged list, and
   under a no-network policy the declaration is ignored and every outbound connection is
   refused. Confirm the declaration covers what your job actually dials.

5. In the `repl` tool, pass `image: r['image']` into
   `host.compute.create('modal', provider_params={...})`. Pass the `im-...` string, never
   an image object.
