# Claude Binder campaign templates

Claude Binder composes campaign and profile JSON files. It validates campaign contracts. It materializes a run bundle. It executes the resulting stage plan.

The module entry point is `python3 -m claude_binder`, and it is the one an installed skill has. A `pip install` of the source distribution also creates a `claude-binder` console script. Copying a skill directory into place does not, because no packaging metadata ships with the skill.

## Included templates

`campaign.template.json` is the base campaign template. Profiles in `profiles/` overlay it. `campaign.example.json` is the PD-L1 companion for `profiles/small-run.template.json`. It keeps every provider endpoint unresolved. `campaign.example.residue-map.json` maps the bundled PD-L1 canary target labels to their cleaned labels. `campaign.example.negative-control.cif` is its negative control, a sequence decoy written by `python3 -m claude_binder.make_negative_control sequence-decoy` from the positive control's own complex at shuffle seed 23. The decoy shares the target chain and the binder backbone with the positive and differs from it in binder sequence, at 0.3056 identity.

Both of the example's controls are placeholders for a real panel. Its positive is an RFdiffusion3 design output, not a known binder, so the example demonstrates the shape of a separating panel rather than a separation. Replace the positive with a literature or experimental complex for the same target and site before reading any winner claim from a run. The gate thresholds carried here were not measured on this target.

`local-contract-test.json` uses the packaged deterministic fixture adapter. `modal-smoke.json` is a provider profile with required values left visible.

The deployed co-folding application is ESMFold2-Fast. The published ESMFold2-Full and Protenix v2 arms are unrun. The `small-run` profile records both deviations and the reason for each.

## Two routes for the small run

`small-run.template.json` and `small-run-modal.template.json` run the same design and scoring shape on different compute. The Modal profile inherits the other one, so every scientific choice is stated once. Pick the route that matches the account you hold.

`small-run.template.json` reaches its three tools over HTTPS deployments and reads their URLs from `provider_endpoints` in the campaign JSON. `small-run-modal.template.json` runs on your own Modal account through the Claude Science job surface and reads no endpoint fields at all, because an endpoint field is enforced only when an adapter command template names its token. It asks instead for a workspace, a spend ceiling, and one environment identity and image reference per adapter group.

The **backbone generator** is the one place the two routes differ. RFdiffusion3 reaches its model over an HTTPS deployment and has no Modal binding in this package, so the Modal route generates with RFdiffusion and leaves `--contigs` at `__REQUIRED__`.

## Supplied candidates on Modal

`supplied-candidates-modal.template.json` inherits the small Modal run and removes its generator and
sequence designer. Set `supplied_candidates.manifest_path` in the campaign to a complete candidate
JSONL manifest. Composition counts the rows. Materialization copies the named FASTA, candidate
structure, and design pose files into the bundle and verifies their hashes. The profile then runs
the same filter, ESMFold2-Fast, scoring, ranking, and report stages.

## Provider endpoints

Set the three endpoint fields to deployments on an account you control. Writing these values declares that you control the deployments. The required-value marker is `__REQUIRED__`, the same marker the target structure and residue map use. `compose` refuses each unresolved endpoint before it writes a configuration.

`data/model-roster.json` says what to deploy. Each row names the application, the source revision, the model revision, the environment identity, and the hardware it was measured on.

`profiles/afm-substitute-two-arm.template.json` preserves an operator-configured AlphaFold2-Multimer-v3 adapter. It does not add a running arm. Before execution it needs an endpoint plus qualified immutable source, weights, image, hardware, and network facts for an application that returns mmCIF and full PAE output.

A campaign that omits `provider_endpoints` can recover them from its own qualified roster. Put the roster at `data/model-roster.json` beside the campaign because `runtime.model_roster_path` resolves relative to the campaign file. `qualify` records the full URL each adapter ran against in `deployment_id`. The packaged roster records application identities without URLs, so it cannot supply an endpoint.

## Bring your own endpoints

The two-arm substitute profile runs over HTTPS deployments rather than on Modal.
Supplying its endpoints is out of scope for a Modal campaign. If you are running that
route, fill every field of `provider_endpoints` with deployments you control.

This object is the smallest complete input for endpoint resolution. A qualified roster can replace it after you run the authorized `qualify` canaries. Do not hand-copy the package roster because its evidence records a different deployment. A runnable replacement roster needs the PASS rows and local evidence that `qualify` writes.

This package ships clients for RFdiffusion3, ProteinMPNN, ESMFold2-Fast, and AlphaFold2-Multimer-v3. The first three have measured application records; the AlphaFold2-Multimer-v3 route is operator-configured rather than deployed. It ships no hosted application sources and no deployment recipe.

## Compose and check

```bash
export BINDER_SKILL="/path/to/installed/claude-binder-lane"

PYTHONPATH="$BINDER_SKILL" python3 -m claude_binder.lane compose \
  --campaign /path/to/campaign.template.json \
  --profile /path/to/profile.json \
  --out /path/to/composed.json

PYTHONPATH="$BINDER_SKILL" python3 -m claude_binder.lane check \
  --config /path/to/composed.json
```

`check` reports unresolved template values. `materialize` requires the resolved values that affect execution.

## Resolve a site from a natural partner

Use `make_target_inputs` before composition when you know a natural binding
partner but do not know the target contact residues. The command queries the
free RCSB PDB APIs, requires a single matching deposited complex, and writes a
target structure, residue map, site block, and resolution artifact.

```bash
export BINDER_SKILL="/path/to/installed/claude-binder-lane"

PYTHONPATH="$BINDER_SKILL" python3 -m claude_binder.make_target_inputs \
  --out generated/target.residue-map.json \
  --site-out generated/target.site.json \
  --target-accession P12345 \
  --partner "Partner name"
```

If RCSB PDB finds several matching complexes, rerun the command with
`--pdb-entry <entry-id>`. The resolver refuses to select an arbitrary
biological state. If no complex is present, it reports the missing pair and
asks you for a verified local complex and its chain IDs.

Copy the generated site block into `targets[].site`. Set
`targets[].structure_path` to the generated reference-complex CIF file and
set the target chain from the artifact. Keep
`site.resolution_artifact_path` relative to the campaign file. Materialization
validates that artifact, copies it to `inputs/site-resolutions/`, and records
its SHA-256 value in the run identity.

The output applies the published target-partner contact rule: any target heavy
atom within 5.0 Å of a partner heavy atom. The resolution artifact records the
structure ID, target and partner chains, experimental or predicted status,
cutoff, residue count, and categorical confidence basis.

## Package data

Use `claude_binder.paths.package_root()` to locate these templates from an installed package. Run outputs belong below the configurable data root. They do not belong in the installed package directory.
