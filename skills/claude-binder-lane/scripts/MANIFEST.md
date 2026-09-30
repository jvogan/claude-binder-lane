# Script manifest

`scripts/` contains the executable sidecars shipped with the Claude Binder skill. This page
records what each one does and which of its paths other components depend on.

## Inventory

| File | Role |
|---|---|
| `bootstrap_environments.py` | Plans environment coverage and, inside a Claude Science kernel, builds explicitly selected environments. It writes results separately and never edits a profile in place. |
| `dispatch_modal.py` | Converts materialized stages into guarded Modal bootstrap, smoke, scale, finalize, receipt, spend, and resume waves. Its CLI only plans and verifies; kernel functions perform submission. |
| `dispatch-policy.template.json` | User-copyable Modal dispatch policy. `volume_name` is intentionally unset. |
| `small-campaign-settings.template.json` | Plan template for a bounded BindCraft2 campaign with controls, two predictor lineages, five seeds, and a shared spend ceiling. |
| `small_campaign.py` | Agent-run commands for plan validation, admission tickets, BindCraft2 shards, prediction workers, artifact indexing, scoring, and ranked cost reports. |
| `small_campaign_esmfold.py` | Stock ESMFold2-Fast worker selected by the explicit `esmfold2-platform` arm. |
| `validate_complexa_proteinmpnn.py` | Independently checks Complexa structures, ProteinMPNN FASTA/NPZ outputs, and exact/off semantic parity. |
| `prepare_proteinmpnn_handoff.py` | Prepares an explicit generated-complex handoff through the pinned native parser, preserves atom and residue lineage, records fixed chains, and verifies sampled target sequences and masks. |
| `validate_genie_px.py` | Parses actual Genie3/PXDesign atom tables, chain roles, kit manifests, and runtime lever evidence. |
| `validate_rfdiffusion3_boltzgen.py` | Checks RFdiffusion3 structures and engagement; verifies BoltzGen generation or full pipeline outputs and original inference defaults. |
| `validation/check_ipsae.py` | Independently recomputes ipSAE and compares it with recorded values. |
| `validation/check_mmseqs2_uniref90.py` | Checks the MMseqs2/UniRef90 sequence-search contract and captured outputs. |
| `validation/check_sc_dockq.py` | Independently recomputes sc_DockQ and compares it with recorded values. |
| `validation/fetch_remote_zip_member.py` | Fetches named members from a remote ZIP64 archive using HTTP range requests. |
| `validation/preflight_adapters.py` | Exercises adapter preflight contracts without provider dispatch. |
| `validation/replay_adapters.py` | Replays recorded adapter inputs and checks declared outputs. |
| `viewer/make_viewer_scripts.py` | Launches the canonical packaged PyMOL and ChimeraX script builder. |
| `viewer/run_readout.py` | Writes the concise per-stage human readout. |

The local smoke-before-scale helper is package data at
`claude_binder/data/helpers/smoke-then-scale.sh`, not a sibling in this directory.

## Load-bearing paths

`dispatch-policy.template.json` uses the installed-module finalizer,
`python -m claude_binder merge-shards`.

`dispatch_modal.py` stages a package subset at `repo_dir/src/claude_binder` on the run Volume.
Every later wave imports that staged package. Moving the destination requires coordinated changes
to bootstrap, job rendering, finalization, and resume validation.

The dispatcher is split by execution surface:

- Planning, verification, bootstrap manifests, and job specifications run from an ordinary shell.
- `submit_wave`, notification collection, reattachment, and `close_wave` require the Claude
  Science kernel's pre-bound `host` object.
- A bootstrap wave uses the first planned stage's `attempt_id`; bootstrap and scientific waves
  share one approval and reservation.

## Runtime dependencies

The dispatcher and environment bootstrapper are standard-library Python. Validation programs may
need the scientific packages they validate. They should fail with the missing dependency named,
not silently substitute a different implementation.

`PYTHONSAFEPATH=1` may remove the implicit script directory from `sys.path`. Entry points that
import siblings must add their own directory explicitly or import through the installed
`claude_binder` package.

## Verification after a change

1. Run `dispatch_modal.py verify` against a mixed local/Modal materialized plan. Only Modal-owned
   stages should be validated; local dependencies should be listed as ignored.
2. Run `dispatch_modal.py dry-run` for the first paid stage and inspect every bootstrap source,
   Volume destination, and digest before authorizing a provider call.

A local run that passes is not a paid live canary. Live verification begins only when the
dispatcher submits, harvests, validates, settles or marks pending, closes every handle, and
survives a resume check on the same run record.
