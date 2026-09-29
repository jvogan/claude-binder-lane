# Troubleshooting

Keep the JSON result and the materialized plan when a Binder command fails. The
message names the immediate missing input, route constraint, or authorization
boundary. Claude Science resolves discoverable configuration and asks you only
when the message requires data, permission, a provider decision, or a scientific
choice.

| Failure class | Meaning | Action |
| --- | --- | --- |
| Target path, chain, residue map, or site error | The configured target cannot be read or does not identify a design site. | Correct the source file or derive the target inputs. Read [target inputs](target-inputs.md). |
| `unresolved required value` or `command token has no value` | The selected Binder route needs an endpoint, image, model revision, account value, or parameter. | Resolve it from the selected provider or profile. If it cannot be resolved, choose a disclosed alternative or use the provider-native route. |
| Adapter is unbound | Binder cannot execute that particular stage. | Use a bound Binder route, a Claude Science native tool, or the tool's documented route. Preserve the handoff artifacts. |
| License or commercial-use refusal | The selected use lacks the terms or agreement the tool requires. | Review [license and commercial use](licence-and-commercial-use.md) for that tool. Change the selected tool or use only an authorized route. |
| Missing ceiling, approval, or estimate | A paid Binder stage has no plan-bound record or active authorization. | Revisit [approval and spend](approval-and-spend.md). Refresh the record from an authorization that covers the revised terms; ask only when they exceed its scope. |
| Provider authentication, quota, or rate-limit failure | The provider rejected the account or request. | Refresh the selected provider authorization, inspect its job log, and resume or retry within the approved plan. Read [provider authorization](provider-authorization.md). |
| Missing provider helper Python or `No module named modal` | The local Claude Science compute integration could not start. This error alone does not establish a Modal service outage. | Check the helper environment for the running platform release. Read [Modal helper recovery](modal-bring-up.md#recover-the-local-provider-helper). |
| `cannot change to` a directory or a missing output path | The command used a path that does not exist in its working directory. | Resolve the checkout, input, and output paths and rerun from the intended directory. Inspect the actual Git error before attributing a failure to sandbox or network restrictions. |
| Missing PAE or an unexpected endpoint response | The selected deployment may not supply an artifact required by the planned score. | Check its recorded schema and returned fields. Use a route that supplies the artifact, or agree on a changed scoring objective. Read [campaign handoffs](running-a-campaign.md#check-handoffs-before-scaling). |
| Modal direct-execution refusal | The selected Modal route needs the guarded dispatcher to receive asynchronous results and retain receipts. | Use [Modal bring-up](modal-bring-up.md#current-paid-execution-boundary). |
| Package drift or missing package file | The installed skill changed during a run. | Re-place the recorded package version, then resume. Do not modify the run bundle. |
| Input contract error between stages | An upstream artifact lacks atoms, an MSA, a pose, or another field the next tool requires. | Fix or replace the upstream artifact, then rerun the affected stage. Read [connector authoring](connector-authoring.md) for a custom bridge. |
| Binder length validation error | The selected minimum and maximum lengths must be positive and ordered. Tool-specific limits come from the selected tool; no package-wide length range applies. | Correct the range or select a tool and route that supports the requested length. |

Do not fabricate a target file, chain, site, endpoint, price, ceiling, model
revision, or score threshold to clear an error. If the error does not identify a
route to resolve it, inspect the selected tool's documentation and record the
actual constraint in the campaign result.
