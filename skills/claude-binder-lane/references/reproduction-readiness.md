# Reproduction readiness

The expanded `rfdiffusion3-two-arm.template.json` profile binds 12 of the
12 published tools. The default `full-ensemble.template.json` binds
6. These counts establish module bindings. They do not establish a
successful provider run, scientific qualification, or exact protocol reproduction.

The report answers three questions separately, because a bound roster is not a runnable
profile. `published_roster_bound` means every roster entry has a bound module.
`profile_configuration_complete` means nothing in the profile is still unset, counting the
`__REQUIRED__` placeholders in adapter rows and the fal deployment endpoints those rows
reference through `{{..._fal_url}}` tokens. `dispatchable_today` is `lane.compose_campaign`
run against `campaign.example.json`, which is the check that actually refuses a run. Each
binding also carries `next_action`, the one thing to do about that row.

None of the three implies another. The default profile composes today and does not bind the
whole roster. The expanded profile binds the whole roster and does not compose, because five
fal deployment endpoints its own commands reference are unset. One boolean stood here until
2026-09-17 and reported the expanded profile as reachable while `lane compose` refused it by
name.

`binding_complete` and `baseline_reachable` remain as the older names for
`published_roster_bound`, with the documented meaning they have always had. They are
deprecated rather than removed, because a reader outside this package relies on them, and
`deprecated_fields` in the report names the replacement. The report also returns
`assessment_scope` and `execution_verification: not_assessed`. None of these fields carries a
claim about a completed run.

The dispatch probe costs local CPU and nothing else. It reads JSON, walks adapter source with
`ast`, and writes one file into a temporary directory it then removes. It opens no socket,
starts no subprocess and calls no provider, so it cannot bill anyone. It takes about nine
seconds on a profile that composes and about twenty milliseconds on one refused at the
endpoint check, because a refusal short-circuits before the stage-contract audit. Pass
`--skip-dispatch-probe` to report `dispatchable_today: null` instead.

`lane.PUBLISHED_BASELINE_TOOL_BINDINGS` records seven generators, two sequence designers
and three co-folding predictors. Reproducing the computational protocol also requires the
published settings, seeds, filters, ensemble construction and scale. Compare those in
[Published campaign comparison](published-campaign-comparison.md). A tool roster alone
cannot settle that comparison, and an unseeded route cannot promise an exact rerun.

## Regenerate the table

The table below describes the expanded profile. Ask about the profile you intend to run:

    PYTHONPATH="$BINDER_SKILL" python3 -B -m claude_binder.reproduction_readiness \
      --profile rfdiffusion3-two-arm.template.json --markdown

Drop `--markdown` for the JSON record, including the per-binding `next_action`, unresolved
operator value paths, required deployment endpoints and cost basis. Omitting `--profile` asks
about the default full-ensemble profile.

## Binding states

| Role | Tool | Adapter ID | Binding state | Module | Operator values | Endpoints | Cost basis |
| --- | --- | --- | --- | --- | --- | --- | --- |
| generators | rfdiffusion | `rfdiffusion-generator` | adapter-shipped | `rfdiffusion_generator` | 4 | - | settled billed amount |
| generators | rfdiffusion3 | `rfdiffusion3-generator` | adapter-shipped | `rfdiffusion3_generator` | 0 | `rfdiffusion3_fal_url` | provider-reported rate |
| generators | freebindcraft | `freebindcraft-generator` | adapter-shipped | `freebindcraft_generator` | 2 | `freebindcraft_fal_url` | unpriced |
| generators | boltzgen | `boltzgen-generator` | adapter-shipped | `boltzgen_generator` | 3 | `boltzgen_fal_url` | unpriced |
| generators | pxdesign | `pxdesign-generator` | adapter-shipped | `pxdesign_generator` | 2 | `pxdesign_fal_url` | unpriced |
| generators | proteina-complexa | `proteina-complexa-generator` | adapter-shipped | `proteina_complexa_generator` | 2 | `proteina_complexa_fal_url` | unpriced |
| generators | genie3 | `genie3-generator` | adapter-shipped | `genie3_generator` | 2 | - | unpriced |
| designers | solublempnn | `solublempnn-designer` | adapter-shipped | `proteinmpnn_designer` | 3 | - | unpriced |
| designers | solublecaliby | `solublecaliby-designer` | adapter-shipped | `solublecaliby_designer` | 6 | - | unpriced |
| predictors | esmfold2 | `esmfold2-predictor` | adapter-shipped | `esmfold2_predictor` | 3 | - | settled billed amount |
| predictors | esmfold2-fast | `esmfold2-fast-predictor` | adapter-shipped | `esmfold2_fast_predictor` | 3 | - | settled billed amount |
| predictors | protenix-v2 | `protenix-v2-predictor` | adapter-shipped | `protenix_v2_predictor` | 2 | - | unpriced |

The three answers the command prints above that table, for this profile:

- **published_roster_bound** true
- **profile_configuration_complete** false
- **dispatchable_today** false (lane.compose_campaign against campaign.example.json)
- **outstanding values** 37 across the published bindings: 32 operator values and 5 deployment endpoints

What each binding state means:

- **adapter-shipped** means a profile selects the published tool and names an existing
  module. Deployment values, model pins and qualification may still be outstanding.
- **contract-only** means a profile row exists but its command names no module.
- **unbound-module** means the module exists but this profile does not bind it to that
  published tool ID. Another profile or a direct invocation may use it.
- **absent** means this profile has no binding and the package has no matching module.

## Next action per binding

Every string below is derived. An unset deployment endpoint carries the gate's own refusal
sentence, so a scientist who acts on it and then runs `lane compose` reads the same words
back. An unset argv element carries the option it supplies and the adapter `--help` that
documents the value, which is the sentence the runtime already composes for its own refusal.
A row with no module names the module file and, when another shipped profile binds it, that
profile.

- `rfdiffusion-generator` Fill `command_argv_template[25]` in the `rfdiffusion-generator` row, the --contigs value for rfdiffusion-generator. Run python3 -m claude_binder.adapters.rfdiffusion_generator run --help to read the value it accepts. 3 more unset values remain in this row.
- `rfdiffusion3-generator` provider_endpoints.rfdiffusion3_fal_url is unresolved; set it to an HTTPS URL for your RFdiffusion3 deployment before composing this profile.
- `freebindcraft-generator` provider_endpoints.freebindcraft_fal_url is unresolved; set it to an HTTPS URL for your FreeBindCraft deployment before composing this profile.
- `boltzgen-generator` provider_endpoints.boltzgen_fal_url is unresolved; set it to an HTTPS URL for your BoltzGen deployment before composing this profile.
- `pxdesign-generator` provider_endpoints.pxdesign_fal_url is unresolved; set it to an HTTPS URL for your PXDesign deployment before composing this profile.
- `proteina-complexa-generator` provider_endpoints.proteina_complexa_fal_url is unresolved; set it to an HTTPS URL for your Proteina-Complexa deployment before composing this profile.
- `genie3-generator` Fill `environment_identity` in the `genie3-generator` row. It reads `modal-env:genie3_generator_gpu@spec_sha=__REQUIRED__` today. 1 more unset value remains in this row.
- `solublempnn-designer` Fill `source_revision` in the `solublempnn-designer` row. It reads `__REQUIRED__` today. 2 more unset values remain in this row.
- `solublecaliby-designer` Fill `command_argv_template[5]` in the `solublecaliby-designer` row, the --caliby-root value for solublecaliby-designer. Run python3 -m claude_binder.adapters.solublecaliby_designer run --help to read the value it accepts. 5 more unset values remain in this row.
- `esmfold2-predictor` Fill `source_revision` in the `esmfold2-predictor` row. It reads `__REQUIRED__` today. 2 more unset values remain in this row.
- `esmfold2-fast-predictor` Fill `source_revision` in the `esmfold2-fast-predictor` row. It reads `__REQUIRED__` today. 2 more unset values remain in this row.
- `protenix-v2-predictor` Fill `model_revision` in the `protenix-v2-predictor` row. It reads `__REQUIRED__` today. 1 more unset value remains in this row.

## What each gap costs

The remaining work is specific to the route and the experiment. A module count cannot
price it. [Tool catalogue](tool-catalogue.md#derive-the-route-matrix) distinguishes platform
skills, hosted APIs, private deployments and self-hosted compute. Claude Science can also
run a tool outside Binder and hand its artifacts into the campaign; the Binder count does
not limit the platform's capabilities.

PXDesign, Proteina-Complexa, BoltzGen and FreeBindCraft use the operator's own fal
application. Their response decoders have offline service-contract tests. No live
qualification of these new package ports is claimed here. Record deployment identity,
weights evidence, an N=1 result and cost evidence before treating a route as qualified.

Proteina's service accepts a contiguous chain-A target and a fixed 64-residue binder.
The adapter preserves the author's target, sends a separately hashed renumbered input,
records a reversible mapping and restores author labels on the returned target pose.
PXDesign checks that an author-numbered hotspot exists before forwarding it. It does
not manufacture a model identity when the service supplies none: the receipt marks that
comparison `not-reported`.

BoltzGen's verified package route accepts the PDB emitted by target preparation.
Its service returns plain base64 CIF/PDB payloads, a sequence and native metrics.
The adapter parses that contract and retains the actual metric object. A missing native
filter verdict stays `unreported`. Its pinned console route has no seed control, so a
requested seed does not establish deterministic generation.

FreeBindCraft publishes a staging set with explicit accepted/rejected provenance.
Missing filter evidence is refused, and response IDs are checked against the request.
Its PyRosetta-bypassed route retains the limits documented in the
[tool catalogue](tool-catalogue.md); a staging row is not an accepted design.

SolubleCaliby's baseline binding uses fixed-backbone design on the RFdiffusion arm.
Its ensemble mode has additional inputs and dependencies and must be selected explicitly.
The fixed-backbone profile does not claim to reproduce the published ensemble experiment.

## What the baseline already has

All three published predictor modules ship. ESMFold2-Full and ESMFold2-Fast share a
model lineage, so running both does not make them independent evidence.
The default profile records one screen seed, five rescore seeds and the published
fifty-scored-candidate floor per generator. A template value still needs validation
against the realized campaign, including controls and any dropped candidates.

The expanded profile has 3 bindings carrying a settled billed cost
basis. Other bindings remain unpriced or modeled; porting a module does not create a
price. Use [Measured costs](measured-costs.md) and the selected route's actual fan-out
when sizing a budget.

## Read the derived states together

`generation.published_generator_requirements.method_binding_states` and its configured
and missing method lists describe the same profile in its declared method order.
They are checked against `claude_binder.reproduction_readiness`. Consult the states and
operator values together, and retain the run's qualification evidence separately.

## The table answers for one profile

`full-ensemble.template.json` binds 6 of 12.
`small-run.template.json` binds 1 of 12.
`rfdiffusion3-two-arm.template.json` binds 12 of 12.
The historical two-arm name now refers to the expanded profile; the default profile
stays smaller so new tools do not silently enter unrelated supplied-candidate campaigns.

An overlay is resolved through the same loader as the executor, so inherited bindings
are included. Profile selection, service compatibility, runtime identity and scientific
qualification remain separate questions even when all twelve modules are present.
