# Optional local diagnostic

Use the local-contract fixture only to diagnose package loading, configuration
composition, materialization, and local graph execution. It runs stand-in
adapters with no account, network, approval, or provider charge. It produces no
prediction or evidence for a biological target.

Claude Science can run this diagnostic when a skill installation or local
execution path is in doubt. A scientist starting a real campaign does not need
to run it.

In a shell outside the installed skill directory, set `BINDER_SKILL` to the
installed skill root and run the diagnostic sequence:

```sh
export BINDER_SKILL="/path/to/claude-binder-lane"
mkdir -p binder-diagnostic && cd binder-diagnostic

PYTHONPATH="$BINDER_SKILL" python3 -B -m claude_binder.lane compose \
  --campaign "$BINDER_SKILL/claude_binder/data/fixtures/local-contract/campaign.json" \
  --profile "$BINDER_SKILL/claude_binder/data/templates/profiles/local-contract-test.json" \
  --out composed.json --json
PYTHONPATH="$BINDER_SKILL" python3 -B -m claude_binder.lane check \
  --config composed.json --check-paths --json
PYTHONPATH="$BINDER_SKILL" python3 -B -m claude_binder.lane materialize \
  --config composed.json --out run-bundle --data-root run-data --json
PYTHONPATH="$BINDER_SKILL" python3 -B -m claude_binder.lane preflight \
  --config run-bundle/config.resolved.json --plan run-bundle/run-plan.json
PYTHONPATH="$BINDER_SKILL" python3 -B -m claude_binder.lane execute \
  --plan run-bundle/run-plan.json \
  --run-root run-data/runs/local-contract-v1 --stage all --json
```

The diagnostic passes when `status.json` reports `state: "completed"` and
`ok: true`, with no provider-facing receipt or nonzero spend record. Do not use
fixture output as a target-specific ranking, tool qualification, cost estimate,
or reproduction result.

If a command fails, keep its JSON result and read [troubleshooting](troubleshooting.md).
For a scientific campaign, start with [getting started](getting-started.md)
instead.
