# The local pipeline implementation

Read this when running the implementation this skill ships. `scripts/MANIFEST.md` lists the skill-owned support files. The installed layout places the package under `claude_binder/`.

## Contents

- [What it is](#what-it-is)
- [What it does not do](#what-it-does-not-do)
- [The three required files](#the-three-required-files)
- [Re-materialize after code changes](#re-materialize-after-code-changes)
- [A sharded stage needs a merge step](#a-sharded-stage-needs-a-merge-step)
- [Numbers the graph settles](#numbers-the-graph-settles)
- [Current differences from the published protocol](#current-differences-from-the-published-protocol)

## What it is

The campaign pipeline ships with this skill as code. It composes and validates campaign configurations, materializes run bundles, executes the graph, and ranks results. The ranking step is a pure function over JSON running on local CPU, so an identical input cohort produces identical output.

The published protocol ranks six terms across three configured predictor modes. The shipped `small-run` profile enables ESMFold2-Fast alone and logs the unrun ESMFold2-Full and Protenix v2 arms. The local fixture graph materializes 44 stages.

Two profiles ship that same small run on different compute. `small-run.template.json` reaches its tools over HTTPS deployments you own, and `compose` refuses it until the campaign fills three URLs in `provider_endpoints`. `small-run-modal.template.json` runs on your own Modal account through the Claude Science job surface, reads no endpoint fields, and asks for a workspace, a spend ceiling, and one environment identity and image reference per adapter group. It generates with RFdiffusion because RFdiffusion3 has no Modal binding here. [Modal bring-up](modal-bring-up.md) lists the calls that produce each Modal value.

## What it does not do

The local executor runs each stage in one subprocess. It validates adapter resource declarations and stage limits, enforces the approval gate, and applies the provider budget cap. Binder submits its Modal route through Claude Science `host.compute` on the user's account. The `remote-compute-modal` skill is an optional runbook and environment source. A qualified user-supplied Modal environment and image use the same host interface without that skill. Binder's approval, reservation, spend-cap, receipt, and cleanup path governs work it submits. The local subprocess route does not allocate the CPU, memory, or GPU limits declared in adapter configurations.

## The three required files

The executor relies on three files for complete local execution:

- `claude_binder/lane.py`
- `claude_binder/stage_contract_check.py` (executed at the terminal validation gate)
- `claude_binder/data/helpers/smoke-then-scale.sh` (used by smoke-scale stages)

The executor resolves the contract checker as `claude_binder.stage_contract_check` and loads the smoke-scale helper from package data. Check all three files before execution.

## Re-materialize after code changes

`materialize` embeds a SHA-256 hash of the runner source in the run identity. Modifying the executor after materialization causes `execute` to refuse the run due to a source hash mismatch. Any edit to the executor invalidates existing bundles on disk.

## A sharded stage needs a merge step

A local stage runs as one synthetic shard. The job register records that stage process with `shard_count: 1` and no provider job ID. Shard manifests from external dispatchers merge with `merge-shards` before terminal validation.

## Numbers the graph settles

The local fixture graph completes 44 stages. The fan-out register records each resolved stage width before executor dispatch. The job register records one local stage process as one synthetic shard.

The executor estimates provider call counts from the configured workload. Read approved stage estimates and the job register together when reviewing workload accounting.

Three output properties require careful handling:

- **Empty error list on failure**: Gate failures reside in the controls and portfolio blocks, and the top-level status is their conjunction. Read the status first, followed by control gates and the portfolio. Process exit codes are non-zero on failure.
- **Non-empty selection on failure**: The controls-failed path empties candidate selection, while the portfolio-failed path retains candidate rows.
- **Surviving portfolio-rank annotations**: Rows refused by the run retain portfolio-rank annotations. Filter by official promotion status instead of scanning for rank fields.

## Current differences from the published protocol

**Ranking scope**: The published protocol uses six terms across three configured predictor modes. The `small-run` profile uses two terms on one mode of ESMFold2.

**Incomplete coverage**: The ranker assigns `-1e308` to candidates that fail seed coverage or filter eligibility. The row retains coverage fields and the final report records missing arm values.

**Contact metrics**: The ranker records `site_contact_iou` and `target_contact_recall` as diagnostics. Default selection tie-breakers evaluate these values after `rank_score`. They do not determine candidate eligibility. Detailed rationale appears in `references/scoring-details.md`.
