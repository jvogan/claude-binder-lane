# New-target provider canary

Use this route when the target is new to the campaign and the immediate question is whether one or more provider paths can fold and score a small supplied panel. It is a bounded route check, not a substitute for target qualification or a binder claim.

## Start from the package surfaces

1. Prepare the target with `python -m claude_binder.make_target_inputs`. Use a deposited complex when one supports the intended site, or supply the site explicitly. Do not recover target logic or provider drivers from historical evidence directories.
2. Build one candidate manifest and reuse it unchanged for every provider arm. Include a deposited cognate binder when available, plus a control whose expected behavior is supported by evidence. A same-scaffold sequence is a challenge control unless a source establishes it as a non-binder.
3. Compose with `supplied-candidates-fal.template.json` or `supplied-candidates-modal.template.json`. These profiles keep generation out of the canary and route the same candidate records through normal validation, folding, parsing, scoring, receipts, and artifact checks.
4. Start at `N=1`. Require one parsed structure observation, declared input and output hashes, terminal provider state, cleanup, and honest reservation or charge accounting before widening. A provider that supplies no settled per-job amount remains recorded as unknown; that alone does not require repeating successful work.

The deposited complex establishes a reference site and a pose-recovery control. It does not establish that a designed sequence binds or works biologically. If the selected predictor has no target-bound validation gate, keep its observations visible as exploratory and do not promote them as production-qualified scores.

Check both contact coverage and construct context. The recorded HER2 windows
372–517 and 365–525 contained every measured epitope residue, yet failed pose
recovery; the wider 320–530 window recovered the reference pose. Contact-span
coverage alone does not establish a suitable folding construct.

## Compare providers without changing the experiment

Call the result a same-model provider comparison only when these values match and are verified in both receipts:

- checkpoint repository and resolved revision;
- MSA inputs and MSA policy;
- inference and recycling settings;
- target and binder sequences;
- target construct, chain convention, and residue numbering;
- seed or the model's documented deterministic setting.

If any item differs, report cross-model corroboration instead. Agreement across different predictors is useful evidence, but it does not isolate provider effects.

## fal route

Use the packaged `esmfold2-fast-predictor` adapter. Set `CLAUDE_BINDER_FAL_CREDENTIAL_ENV` to the name of the host-managed variable that contains the fal credential. Do not copy, print, or write the value. The authorization check and runtime must resolve the same credential route.

The adapter calls `hydrate` once before its first fold and refuses to predict
unless the application reports complete. Use `--skip-hydrate` only when the
application is already known warm. The stage admission estimate still carries
the recorded cold-load upper bound; hydration is not treated as free.

A 401 means the selected process credential was rejected. A 403 means that credential lacks access to the application. Neither result proves that the application is absent. Probe only the application selected by the plan.

`max_seconds` is the maximum duration requested from this application. It is not a published fal platform limit and is not a biological sequence-length rule. `timeout_seconds` is the caller's wall-clock supervision bound. Record cold or warm state, total residues, model revision, and elapsed time before inferring a runtime relationship.

## Modal route

Read the live environment ledger and workspace identity in Claude Science, then bind closures over those returned values as described in [Provider authorization](provider-authorization.md). Resolve the environment image and Volume mapping from that ledger. Do not assume an account-specific cache name.

An offline image must mount the cache that contains the pinned model revision. It must not silently fetch a current revision. Run the materialized plan through the guarded `dispatch_modal.py` workflow. Historical scripts that call `host.compute.create` directly are inspection records, not the supported campaign transaction.

## Stop and widen rules

After the first prediction, verify the parser receipt and structure bytes before starting scale. On Modal, close a terminal handle or reattach an accepted job before retrying it. Before each fal prediction, the adapter writes a stable exact-call identity to a hash-chained journal under the run artifact root. A timeout leaves that call unknown across new attempts and changed local wall limits until operator evidence reconciles it. Unknown state for one call does not block unrelated providers or candidates.

Reuse the user's current provider choice and spend ceiling for retries and bounded widening inside the same stated scope. Ask again only when the provider, scientific scope, or ceiling changes.

Completed fal calls record their response and parser-file locations relative to
the journal. A new attempt verifies and restores those files before parsing,
without repeating hydration or prediction. Keep the journal and original attempt outputs
together when moving a run. Journals written before file locations were recorded
can reuse verified files already restored to the requested attempt directory.

Source: [`make_target_inputs.py`](../claude_binder/make_target_inputs.py) prepares target inputs. [`tool_menu.py`](../claude_binder/tool_menu.py) advertises this workflow. [`fal_esmfold2_fast_predictor.py`](../claude_binder/adapters/fal_esmfold2_fast_predictor.py) owns the fal route. [`dispatch_modal.py`](../scripts/dispatch_modal.py) owns guarded Modal dispatch.
