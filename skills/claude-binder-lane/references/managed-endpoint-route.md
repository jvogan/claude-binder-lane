# Managed model endpoint route

Claude Science can register and call a model service through its native endpoint
surface. The scientist configures a reachable service and its runbook, then uses
the platform's call contract to address that registered endpoint.

**This package binds no managed endpoint today.** Claude Science can run a
registered model standalone, including a separately configured NVIDIA BioNeMo
NIM, and hand hashed artifacts and provenance into a Binder campaign. Automated
Binder dispatch additionally requires an adapter and its job lifecycle.

Registration handles the platform's credential and transport indirection. It
does not supply every model in a family or establish availability for a session.
Each selected model needs its own request schema, response parser and runbook.
A campaign also needs the artifact and provenance contract its downstream stage
reads. Read [connector authoring](connector-authoring.md) and
[what a campaign records](#what-a-campaign-must-record) before treating a result
as campaign evidence.

Source for every statement below, read on 2026-09-12 from the installed runtime:
`managed-model-endpoints/SKILL.md` and `using-model-endpoint/SKILL.md`,
`provider.json` and `provider.py`, transcribed in
[the platform snapshot](evidence/claude-science-platform-skills-0.1.47.md) at
`0.1.47-release`. Both skill files are byte-identical from `0.1.41-release`
through `0.1.47-release`, so this contract has not moved across six releases.

## The two halves

`managed-model-endpoints` is the registration contract. `using-model-endpoint`
is the call contract. They are separate skills and a campaign uses both.

## Enablement is once per machine, and the mode is exclusive

The scientist connects the family under Customize, Compute, Model endpoints. The
setup flow saves the family credential first, and connecting without a key is not
a state the platform offers.

One mode is picked per machine. **Local** registers container services the daemon
starts and stops. **URL or remote** registers an https endpoint against the
configured host. Until the family is connected, `free_port()` and `register()`
raise a precise error, and in the wrong mode they refuse and name the setting.
Endpoints already registered under the other mode keep dispatching. Only new
registrations refuse.

That exclusivity is the same fact the
[BioNeMo NIM route](bionemo-nim-route.md) records from a Claude Science run on
2026-08-31. Mixing a local NIM and a hosted NIM takes one host per mode.

Disconnecting is a full teardown. Every local service is stopped through its
approved stop script and every registration is removed, local and hosted. Caches
stay on disk. A stop that fails keeps that one row, marked failed.

## Registration

```python
port = host.model_endpoints.free_port()        # local only, random 20000-29999

host.model_endpoints.register(
    name="boltz2-service",            # <model>-service, never the bare model name
    url=f"http://127.0.0.1:{port}",   # literal 127.0.0.1, localhost is rejected
    credential="NVIDIA_API_KEY",      # the credential NAME, never a value
    skill="<model-runbook-skill>",
    start=START_SCRIPT,
    stop="docker stop boltz2-service",# exit 0 only once actually stopped
    live="/v1/health/ready",          # 200 means the model answers
)
```

Five rules carry consequences for a campaign.

**The credential name is fixed.** Every registration passes
`credential="NVIDIA_API_KEY"` and the daemon rejects any other name. This is a
platform rule, not a convention. For a local service the value feeds the start
script's registry login and never enters the kernel environment. For a remote
endpoint it authenticates the upstream and is delivered only into the inference
cell, as `INFER_API_KEY`.

**The readiness route must distinguish loading from ready.** A service that is up
but still loading weights has to answer not-ready on `live`, or the daemon will
run a cell against a model that cannot answer it.

**The stop script must not exit 0 early.** It exits 0 only once the service is
actually stopped, because the daemon swaps one resident model off the device
before starting the next.

**Registration always cards the user**, showing the scripts verbatim, the port,
the service directory and the credential name. A byte-identical re-registration
is silent. Any byte change cards again.

**Container specifics are not the platform's.** Image, registry login, internal
port, cache mount target and readiness route come from the model's own runbook
skill, named in the registration's `skill` field.

## Calling a registered endpoint

The only dispatch form is the `compute_provider` tool with the registered name.

```python
compute_provider(provider="boltz2-service", code="""
import requests
r = requests.post(BASE_URL + "/v1/infer", json=payload)
""")
```

`BASE_URL` is preloaded, both as a Python variable and as
`os.environ["BASE_URL"]`. Build every request URL from it. A hardcoded host or
port does not reach the endpoint, because the kernel's network egress is scoped
to exactly one host, declared in `provider.json` as the control host on port 443.

Hosted endpoints take `Authorization: Bearer $INFER_API_KEY`. Local endpoints
need no auth header. `httpx` is preinstalled and pinned at `0.28.1` in the helper
environment, and `requests` works too.

Requests ride the sandbox HTTP proxy. `HTTP_PROXY` and `HTTPS_PROXY` are set, and
disabling them makes the endpoint unreachable. Passing `trust_env=False` to an
httpx client is the common way to break this.

An endpoint is not a kernel environment. Passing `environment="boltz2-service"`
to a plain Python cell fails, and a plain cell gets no `BASE_URL`.

A first cold start downloads the image and the weights and can take minutes. The
daemon streams that progress into the cell.

## There is no job lifecycle on this route

`provider.py` refuses `create_sandbox`, `exec`, `list_owned`, `read_owner` and
`terminate`, with the message that inference providers have no job lifecycle. The
call is a direct request and response.

This is the sharpest constraint for this package, and it is why a managed endpoint
cannot reuse a job-dispatch shape. A configured job route can record an identifier,
attach after a crash, harvest a receipt, settle spend, and clean up. Modal and fal
ship such Binder routes; RunPod needs its worker contract brought up, and Lambda
needs a provider transport. A managed endpoint offers none of that. It returns no
job identifier, so a crash leaves nothing to resume and no provider receipt to
validate.

A Binder adapter for this route therefore has to carry its own evidence. That is
the bring-up work, and it is not done.

## The NVIDIA credential is first class

`provider.py` declares `secret_env_prefixes = ("INFER_", "NVIDIA_")` and scrubs
tokens matching `nvapi-` from output. The registration contract requires the name
`NVIDIA_API_KEY`. The platform's inference provider is built around NVIDIA
credentials, so an NVIDIA BioNeMo NIM is a supported endpoint for this family
rather than an adaptation of it.

That says nothing about which NIM endpoints a given account can reach, what they
cost, or whether their outputs parse. Read the
[BioNeMo NIM route](bionemo-nim-route.md) for the atomic tools, the MSA topology
and the artifact handoff, and for the measured note that a hosted NIM exposes no
free health check, so its first evidence is the N=1 call.

## Measured on this account

Read from the Claude Science org database on 2026-09-12, opened read-only. These are
facts about one account and they do not travel to another one. Read them as proof that
the route works and as a source for request shapes, never as a rate card.

Five endpoints are registered in the `infer` family, all hosted, all enabled, each with
a credential name and an approved-script hash.

| Registered name | Endpoint | Runbook skill it names |
| --- | --- | --- |
| `msa-search-service` | `health.api.nvidia.com/v1/biology/colabfold/msa-search` | `alphafold2` |
| `openfold3-service` | `health.api.nvidia.com/v1/biology/openfold/openfold3` | `openfold3` |
| `boltz2-service` | `health.api.nvidia.com/v1/biology/mit/boltz2` | `boltz` |
| `rfdiffusion-service` | `health.api.nvidia.com/v1/biology/ipd/rfdiffusion/generate` | `claude-binder-lane` |
| `proteinmpnn-service` | `health.api.nvidia.com/v1/biology/ipd/proteinmpnn/predict` | `proteinmpnn` |

The `managed_endpoints` table is empty and these live in `compute_providers`, so the
family is in URL and remote mode rather than holding daemon-managed local containers.
The modes are exclusive, so a local container registration would refuse while these
stand.

Forty-five calls have completed. OpenFold3 16, Boltz-2 14, MSA search 6, ProteinMPNN 5,
RFdiffusion 4, against 6 failures in total. The first was on 2026-09-01 and the most
recent on 2026-09-11. Durations run from 1 second to 247 seconds.

A complete generate-then-design chain ran on 2026-09-05 between 21:20 and 21:27:
RFdiffusion at 3, 22, 86 and 64 seconds, then ProteinMPNN at 6, 8, 5, 35 and 14 seconds.
The target was GFP, chain A residues 3 to 229, a 60-residue binder, hotspots A146, A174,
A206 and A221.

**No cost evidence exists.** Every completed row records `{"status":"done","exit_code":0}`
and an empty hardware field. The ledger holds wall clock and nothing else. A NIM arm is
therefore unpriced, `qualify` cannot quote it, and `budget_plan` refuses to size a plan
against it. Record a rate from NVIDIA's own billing before treating this route as priced.

Unpriced is not free. [Measured costs](measured-costs.md) defines an unpriced row as one where the run happened and no charge was recorded against it, which is a statement about this ledger rather than about the vendor's invoice. Nothing here establishes that a call on this route costs nothing. Treat a NIM arm as a cloud dispatch of unknown cost: it reaches an external model service, so it needs the same authorization any other provider call needs, and the absence of a dollar figure is not the authorization. Decision 7's silent case withholds paid dispatch; it does not convert an unpriced route into a free one.

### Request shapes that returned 200

These came from the cells that made those calls, so they are read rather than inferred.
They are the shapes NVIDIA accepted on those dates, and NVIDIA can change an API.

```python
# rfdiffusion-service
{"input_pdb": pdb, "contigs": "A3-229/0 60-60",
 "hotspot_res": ["A146", "A174", "A206", "A221"], "diffusion_steps": 50}

# openfold3-service and boltz2-service
{"polymers": [{"id": "A", "molecule_type": "protein", "sequence": seq}],
 "recycling_steps": 1, "sampling_steps": 30, "diffusion_samples": 1}

# msa-search-service
{"sequence": seq, "e_value": 0.0001, "iterations": 1,
 "databases": ["Uniref30_2302"], "output_alignment_formats": ["a3m"],
 "max_msa_sequences": 200}
```

Every call sent `Authorization: Bearer $INFER_API_KEY` with `Content-Type` and `Accept`
set to `application/json`, and posted to `BASE_URL`.

### An adapter for this route can be stdlib-only

The ProteinMPNN cell and the probe cells call the endpoint with `urllib.request`, not
with `httpx`. `httpx` is pinned in the helper environment and it is not required. Binder's
executor is stdlib-only, so an adapter for this route fits that constraint with no new
dependency. That removes the usual reason a hosted route needs a vendored client.

## The route returns no PAE, so it cannot rank the published way

The hosted Boltz-2 endpoint returns `pae: null`, observed on two calls against the Cas9 RNP on 2026-09-16. ipSAE is computed from PAE, so it is not computable on this route.

That matters more than it sounds. ipSAE_min is the published protocol's primary ranking term, weighted 4 against 1 for each sc_DockQ term, so a campaign on this route can produce structures, ipTM, pLDDT and sc_DockQ, but not the published ranking formula. Treat the managed-endpoint route as a screen that yields poses and pose agreement, and plan a different arm for any ranking that has to carry ipSAE. A self-hosted Boltz arm writing full PAE, or Protenix v2 with `--need_atom_confidence true`, both produce it, and both are paid GPU rather than this route.

Check the response for a PAE field before building a ranking on an endpoint, rather than after scoring a pool.

## What a campaign must record

A registered endpoint is a route, not a result. Record all of the following
before a number from it enters a candidate ranking.

| Field | Why |
| --- | --- |
| Endpoint name and registration mode | Two modes cannot coexist on one host, so the mode explains what else was reachable |
| Model and service revision | `BASE_URL` names a service, not a version |
| The runbook skill named in `skill` | It is where the request shape came from |
| Request schema and response fields parsed | The route returns whatever the model returns |
| Price source and the measured cost | Nothing on this route reports a settled bill |
| Egress decision | The kernel reaches one host and the campaign should say which |
| The N=1 artifact that parsed | A response with status 200 is not a parsed artifact |

## What does not transfer

A measurement taken through a managed endpoint prices that endpoint. It does not
transfer to self-hosting the same model on Modal, RunPod or Lambda Cloud, and it
does not transfer to the same model behind a different vendor's endpoint. This is
the same non-transfer rule the [tool catalogue](tool-catalogue.md#route-shapes)
states for hosted routes, and it applies here for the same reason. The run
exercised somebody else's build on somebody else's hardware.

## Bring-up, in order

1. Connect the family and pick the mode. Record which mode.
2. Register the endpoint. Keep the approval card's contents with the campaign.
3. Call it once, at N=1, against the real target input.
4. Parse the response into the artifact a downstream stage reads. Counting a
   status code is not parsing.
5. Record the cost from the provider's own billing, not from an estimate.
6. Only then scale, and only then treat the arm as a campaign route.
