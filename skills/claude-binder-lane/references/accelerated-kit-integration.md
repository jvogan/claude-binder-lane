# Integrate accelerated tools into a campaign

Claude Science can prepare the selected Anthropic kit on the scientist's own
compute and connect it to their existing workflow. Select tools and stages for
the scientific question. A fixed Binder profile is optional. A missing packaged
binding calls for a native run and an artifact handoff, or a campaign-local
connector when automated Binder execution is useful.

## Select the kit and preserve its contract

| Tool | Account-local Modal setup | Modes | Handoff |
| --- | --- | --- | --- |
| Genie3 | [Recipe and qualification](genie3-kit-modal.md) | `off`, `exact`, `fast` | Inspect the C-alpha binder backbone and select a compatible sequence designer. |
| PXDesign | [Recipe and qualification](pxdesign-kit-modal.md) | `off`, `exact`, `fast`, `big` | Preserve target/binder chain roles when passing structures to sequence design. |
| RFdiffusion3 | [Recipe and qualification](rfdiffusion3-kit-modal.md) | `off`, `exact`, `fast` | Decompress and parse real design structures before sequence design. |
| BoltzGen | [Recipe and qualification](boltzgen-kit-modal.md) | `off`, `exact`, `fast`, `big` | Select the requested generation or full design/sequence/refolding workflow. |
| Proteina-Complexa | [Recipe and qualification](complexa-kit-modal.md) | `off`, `exact`, `fast`, `big` | Prepare the reward-model dependencies when that workflow is selected. |
| Full-backbone ProteinMPNN | [Recipe and qualification](proteinmpnn-kit-modal.md) | `off`, `exact` | Emit actual binder sequences, retaining fixed target chains and backbone identity. |

For predictor kits, read [Boltz-2 setup](accelerated-upstream-builds.md#what-a-claude-science-user-can-do-with-this-today)
and [ESMFold2 setup](esmfold2-kit-modal.md). The
[release overview](accelerated-upstream-builds.md) records the remaining kits,
pins and measured upstream claims. A documented upstream route remains available
even when this package does not supply a recipe for it.

Kit `exact` describes equality against that kit's pinned stock tool. Some kit
pins differ from the older catalogue and published binder protocol. Preserve
the intended reproduction claim and record those revision differences; a mode
name alone does not establish reproduction of the published campaign.

Read the [dated native Modal qualification](accelerated-kit-qualification-2026-09-30.md)
for the hardware, modes, outputs and execution scope actually tested.

Do not replace a requested tool, mode, sampler, constraint, batch, or hardware
requirement to make a run pass. Explain an actual incompatibility and retain
the requested route while preparing its missing requirements. Keep the tools'
native interfaces available, including capabilities outside the accelerated
subset. For example, ProteinMPNN `exact` refuses C-alpha-only inputs; Genie3's
C-alpha output requires its compatible checkpoint and route. An explicit
stock or native choice remains available for that handoff.

## Prepare the user's selected route

Discover the account and compute available in the scientist's session. Read
the selected recipe and upstream runbook. Reuse a matching image and verified
weight cache, or prepare the image and hydration within the existing approval.
Account identities, provider image IDs, local paths and credentials belong to
the user's runtime configuration. Derive them from the actual account.

Stage a shipped recipe from the Binder kernel:

```python
staged = binder_stage_environment_recipe("genie3_kit_gpu", workspace)
```

Then use the selected provider's setup in its compute-provider kernel:

```python
built = build_env("genie3_kit_gpu", path=staged["path"], hydrate=True)
```

`workspace` is the session's actual workspace. Staging copies a recipe and
records its hash; building and hydration use provider resources. Read the
environment ledger after building and carry its identity, mounts, runtime
environment and selected card into the job. Use the provider's normal
submission/completion/artifact-retrieval interface. A native Modal controller
can use the same recipe's `build()` return values, `HYDRATE`, and `CHECK`.
[Modal setup](modal-bring-up.md) explains the platform-specific interfaces.

Every real kit invocation names the requested mode. Use the kit's launcher or
documented native activation interface, including RFdiffusion3's separate
overlay interpreter and ProteinMPNN's explicit wrapper. Setting an arbitrary
environment variable around a stock process is insufficient.

## Qualify the handoff, then scale

Carry the smallest input that exercises the selected contract through actual
inference and the next consumer. Preserve full scientific inference settings.
Respect model-specific minimum shapes: BoltzGen's Exact and Fast diffusion
paths require a batch of at least two. A mode check alone establishes setup.

Require a completed invocation, matching activation, the kit's own manifest
and counters where provided, and the declared outputs. Capture stdout and
stderr and pass them with the actual exit code to
`claude_binder.kit_engagement.parse_engagement`. An accepted log verdict is one
part of qualification; independently parse structures or sequences and verify
the requested count, candidate identifiers, chain roles and hashes.

Join stages through explicit artifacts. Record backbone identity and residue
mapping at sequence design, then retain the actual designed sequence and target
identity for complex prediction. Choose predictors and controls for the user's
scoring objective. Generators do not all produce PAE or the same file format.
Confirm the selected predictor supplies the score inputs before expanding a
batch. Read [connector authoring](connector-authoring.md) when the handoff needs
a reusable automated bridge.

Record the tool and kit commits, weight digests, mode, seed, full parameters,
image identity, hardware, input/output hashes, parsed counts, timing and cost
evidence. Use these separately observable states in reports:

- Recipe shipped: reproducible setup is packaged.
- Native Modal tested: the named card, mode, input and output contract ran.
- Claude Science tested: the installed skill exercised its platform integration.
- Binder adapter available: an executable graph contract is packaged.

Untested configurations call for a focused qualification within the authorized
scope. They do not make a documented tool unavailable. A native Modal test
does not assert that Claude Science's submission or account setup was tested.
Keep inference qualification, throughput measurement, full pipeline execution
and scientific validation as distinct claims.
