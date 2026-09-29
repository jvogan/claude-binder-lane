# Claude Binder runbook

## Compose

```bash
claude-binder compose \
  --campaign /path/to/campaign.json \
  --profile /path/to/profile.json \
  --out /path/to/composed.json
```

## Validate

```bash
claude-binder check --config /path/to/composed.json
```

Resolve every reported required value before materialization.

## Materialize

```bash
claude-binder materialize \
  --config /path/to/composed.json \
  --out /path/to/run-bundle \
  --data-root /path/to/run-data
```

The bundle stores the resolved configuration, stage contract, run identity, and selected support hashes.

## Execute

```bash
claude-binder execute \
  --plan /path/to/run-bundle/run-plan.json \
  --run-root /path/to/run-data/runs/example
```

Use `--resume` to continue a compatible interrupted run. The package validates its recorded support identity before running stages.
